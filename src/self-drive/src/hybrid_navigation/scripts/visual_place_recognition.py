#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import BallTree
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool, Float32, Int32, String
from nav_msgs.msg import Odometry
from hybrid_navigation.srv import LoadTarget

class VisualPlaceRecognition:
    def __init__(self):
        rospy.init_node('visual_place_recognition')
        
        # Parameters
        self.descriptor_dim = 512  # Default for ResNet18
        self.model_name = rospy.get_param('~model_name', 'resnet18')
        self.database_path = rospy.get_param('~database_path', 'vpr_database.pkl')
        self.matches_dir = rospy.get_param('~matches_dir', os.path.join(os.path.dirname(self.database_path), 'matches'))
        self.best_match_threshold = rospy.get_param('~best_match_threshold', 0.75)
        
        # Create matches directory if it doesn't exist
        if not os.path.exists(self.matches_dir):
            os.makedirs(self.matches_dir)
            rospy.loginfo(f"Created matches directory: {self.matches_dir}")
        
        # Initialize CNN model for feature extraction
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize database
        self.descriptors = []
        self.odometry_data = []
        self.database_built = False
        self.tree = None
        
        # For best match tracking
        self.best_match_similarity = -1.0
        self.best_match_image = None
        self.best_match_path = None
        
        # Publishers and subscribers
        self.match_pub = rospy.Publisher('/vpr/match', Bool, queue_size=10)
        self.similarity_pub = rospy.Publisher('/vpr/similarity', Float32, queue_size=10)
        self.match_index_pub = rospy.Publisher('/vpr/match_index', Int32, queue_size=10)
        self.best_match_path_pub = rospy.Publisher('/vpr/best_match_path', String, queue_size=10)
        
        # Subscribe to image and odometry
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Target image handling
        self.target_image = None
        self.target_descriptor = None
        self.load_target_image = rospy.Service('/vpr/load_target', LoadTarget, self.load_target_callback)
        
        # Add a subscriber to listen for save requests
        self.save_request_sub = rospy.Subscriber('/topological_map/save_request', Int32, self.save_request_callback)
        
        # Signal handler for saving the database on shutdown
        rospy.on_shutdown(self.save_database)
        
        rospy.loginfo(f"Visual Place Recognition node initialized with best match threshold: {self.best_match_threshold}")
        
    def load_model(self):
        """Load pre-trained CNN model for feature extraction"""
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            # Remove the final fully connected layer
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        else:
            rospy.logwarn(f"Model {self.model_name} not supported, using ResNet18")
            model = models.resnet18(pretrained=True)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        
        model.eval()  # Set to evaluation mode
        return model
    
    def extract_features(self, img):
        """Extract features from image using CNN"""
        try:
            # Preprocess image
            tensor = self.transform(img).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.model(tensor)
            
            # Reshape and normalize features
            features = features.squeeze().cpu().numpy()
            features = features / np.linalg.norm(features)
            
            return features
        except Exception as e:
            rospy.logerr(f"Error extracting features: {e}")
            return None
    
    def image_callback(self, msg):
        """Process incoming images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Extract features from current image
            features = self.extract_features(cv_image)
            
            # If we have a target descriptor, compute similarity
            if self.target_descriptor is not None:
                similarity = np.dot(features, self.target_descriptor)
                self.similarity_pub.publish(Float32(similarity))
                
                # Publish match based on threshold
                is_match = similarity >= self.best_match_threshold
                self.match_pub.publish(Bool(is_match))
                
                if is_match:
                    rospy.loginfo(f"Match found with similarity: {similarity}")
                else:
                    rospy.loginfo(f"Similarity score: {similarity} (below threshold {self.best_match_threshold})")
                
                # Check if this is the best match so far and exceeds the threshold
                if similarity > self.best_match_similarity and similarity >= self.best_match_threshold:
                    self.best_match_similarity = similarity
                    self.best_match_image = cv_image.copy()
                    
                    # Save the best match image
                    timestamp = rospy.Time.now().to_nsec()
                    self.best_match_path = os.path.join(self.matches_dir, f"best_match_{timestamp}.jpg")
                    cv2.imwrite(self.best_match_path, self.best_match_image)
                    
                    # Publish the path to the best match image
                    self.best_match_path_pub.publish(String(self.best_match_path))
                    rospy.loginfo(f"New best match saved: {self.best_match_path} (similarity: {similarity})")
            
            # If we're building the database, add the features
            if not self.database_built:
                self.descriptors.append(features)
                
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")
    
    def odom_callback(self, msg):
        """Store odometry data associated with images"""
        if not self.database_built:
            # Store position and orientation
            pose = msg.pose.pose
            position = (pose.position.x, pose.position.y, pose.position.z)
            orientation = (pose.orientation.x, pose.orientation.y, 
                           pose.orientation.z, pose.orientation.w)
            
            self.odometry_data.append((position, orientation))
    
    def build_database(self):
        """Build a searchable database of descriptors"""
        if len(self.descriptors) == 0:
            rospy.logwarn("No descriptors to build database")
            return False
        
        try:
            # Convert to numpy array
            X = np.array(self.descriptors)
            
            # Build BallTree for efficient similarity search
            self.tree = BallTree(X, metric='euclidean')
            
            # Save database to file
            database = {
                'descriptors': self.descriptors,
                'odometry': self.odometry_data
            }
            
            # Ensure the directory exists
            db_dir = os.path.dirname(self.database_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Save the database
            with open(self.database_path, 'wb') as f:
                pickle.dump(database, f)
            
            self.database_built = True
            rospy.loginfo(f"Database built with {len(self.descriptors)} entries and saved to {self.database_path}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error building database: {e}")
            return False
    
    def save_database(self):
        """Save the VPR database to disk"""
        return self.build_database()
    
    def load_database(self):
        """Load a previously saved database"""
        if not os.path.exists(self.database_path):
            rospy.logwarn(f"Database file not found: {self.database_path}")
            return False
        
        try:
            with open(self.database_path, 'rb') as f:
                database = pickle.load(f)
            
            self.descriptors = database['descriptors']
            self.odometry_data = database['odometry']
            
            # Build BallTree for efficient similarity search
            X = np.array(self.descriptors)
            self.tree = BallTree(X, metric='euclidean')
            
            self.database_built = True
            rospy.loginfo(f"Database loaded with {len(self.descriptors)} entries")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error loading database: {e}")
            return False
    
    def load_target_callback(self, req):
        """Load target image and compute its descriptor"""
        try:
            # Load target image
            target_path = req.image_path
            self.target_image = cv2.imread(target_path)
            
            if self.target_image is None:
                rospy.logerr(f"Failed to load target image: {target_path}")
                return False
            
            # Extract features
            self.target_descriptor = self.extract_features(self.target_image)
            
            if self.target_descriptor is None:
                rospy.logerr("Failed to extract features from target image")
                return False
            
            # Reset best match tracking when loading a new target
            self.best_match_similarity = -1.0
            self.best_match_image = None
            self.best_match_path = None
            
            # If we have a database, find the closest match
            if self.database_built and self.tree is not None:
                dist, ind = self.tree.query(self.target_descriptor.reshape(1, -1), k=1)
                match_index = ind[0][0]
                self.match_index_pub.publish(Int32(match_index))
                rospy.loginfo(f"Closest match in database: index {match_index}")
            
            rospy.loginfo(f"Target image loaded: {target_path}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error loading target image: {e}")
            return False
    
    def save_best_match_with_target(self):
        """Save the best match image side-by-side with the target image"""
        if self.best_match_image is not None and self.target_image is not None:
            try:
                # Resize images to the same height
                h1, w1 = self.best_match_image.shape[:2]
                h2, w2 = self.target_image.shape[:2]
                
                # Use the minimum height of both images
                h = min(h1, h2)
                
                # Calculate new widths while maintaining aspect ratio
                w1_new = int(w1 * (h / h1))
                w2_new = int(w2 * (h / h2))
                
                # Resize both images
                best_match_resized = cv2.resize(self.best_match_image, (w1_new, h))
                target_resized = cv2.resize(self.target_image, (w2_new, h))
                
                # Create side-by-side comparison
                comparison = np.hstack((target_resized, best_match_resized))
                
                # Save the comparison image
                timestamp = rospy.Time.now().to_nsec()
                comparison_path = os.path.join(self.matches_dir, f"comparison_{timestamp}.jpg")
                cv2.imwrite(comparison_path, comparison)
                
                rospy.loginfo(f"Saved comparison image: {comparison_path}")
                return comparison_path
            except Exception as e:
                rospy.logerr(f"Error saving comparison image: {e}")
                return None
        return None
    
    def save_request_callback(self, msg):
        """Handle a request to save the database"""
        rospy.loginfo("Received request to save the VPR database")
        success = self.build_database()
        if success:
            rospy.loginfo(f"VPR database saved successfully to {self.database_path}")
        else:
            rospy.logwarn("Failed to save VPR database")
    
    def run(self):
        """Main loop"""
        # If database exists, load it
        if os.path.exists(self.database_path):
            self.load_database()
        else:
            rospy.loginfo("No existing database found. Will build one from incoming images.")
        
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            rate.sleep()
            
        # Save a comparison image before shutdown
        if self.best_match_path is not None:
            self.save_best_match_with_target()

if __name__ == '__main__':
    try:
        vpr = VisualPlaceRecognition()
        vpr.run()
    except rospy.ROSInterruptException:
        pass 