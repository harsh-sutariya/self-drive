#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import pickle
import rosbag
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32, Time
import torch
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image as PILImage

class Dinov2VLADProcessor:
    def __init__(self):
        rospy.init_node('dinov2_vlad_processor')

        # Initialize publishers
        self.status = rospy.Publisher('/match_status', Bool, queue_size=10)
        self.timestamp_pub = rospy.Publisher('/goal_timestamp', Time, queue_size=10)
        self.goal_id = rospy.Publisher('/match_goal_id', Int32, queue_size=10)
        
        # Load parameters
        self.codebook_path = rospy.get_param('~codebook_path', 'dual_vlad.pkl')
        target_image_path = rospy.get_param('~target_image_path', 'target.jpg')
        self.bag_path = rospy.get_param('~bag_file_path', 'images.bag')
        self.yolo_model_path = rospy.get_param('~yolo_model_path', 'runs/detect/train/weights/best.pt')
        self.num_clusters = rospy.get_param('~num_clusters', 128)
        self.threshold = rospy.get_param('~distance_threshold', 0.8)

        # Initialize models and processors
        self.yolo = YOLO(self.yolo_model_path)
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])

        # Load or generate VLAD codebook
        if not self.load_codebook():
            rospy.loginfo("No codebook found. Generating new codebook...")
            self.generate_codebook()
            self.save_codebook()

        # Build VLAD database and search tree
        self.database = []
        self.timestamps = []
        self.crop_indices = []  # Store which crop from which image
        self.build_vlad_database()

        # Process target image and find goal index
        self.target_image = cv2.imread(target_image_path)
        if self.target_image is None:
            raise FileNotFoundError(f"Target image not found at {target_image_path}")
        
        # Process target image with YOLO
        target_results = self.yolo.predict(source=self.target_image, stream=True)
        target_crops = []
        for result in target_results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                crop = self.target_image[y1:y2, x1:x2]
                target_crops.append(crop)

        if not target_crops:
            raise RuntimeError("No valid crops found in target image")

        # Find best match among target crops
        best_match = None
        best_score = -1
        for crop in target_crops:
            features = self.extract_features(crop)
            query_global = self._compute_vlad(features['global'], self.global_kmeans)
            query_local = self._compute_vlad(features['local'], self.local_kmeans)
            query_vec = np.concatenate([query_global, query_local])
            
            # Find best match for this crop
            similarities = cosine_similarity([query_vec], self.database)[0]
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]
            
            if max_sim > best_score:
                best_score = max_sim
                best_match = max_sim_idx

        if best_match is None or best_score < self.threshold:
            raise RuntimeError("No good match found for target image")

        self.goal_index = best_match
        self.goal_timestamp = self.timestamps[self.goal_index]
        
        rospy.loginfo(f"Goal index: {self.goal_index}, Score: {best_score:.3f}")

        self.status.publish(True)
        self.goal_id.publish(self.goal_index)
        self.timestamp_pub.publish(Time(self.goal_timestamp))

    def load_codebook(self):
        if os.path.exists(self.codebook_path):
            try:
                with open(self.codebook_path, 'rb') as f:
                    data = pickle.load(f)
                    self.global_kmeans = data['global_kmeans']
                    self.local_kmeans = data['local_kmeans']
                rospy.loginfo("Codebook loaded successfully.")
                return True
            except Exception as e:
                rospy.logwarn(f"Failed to load codebook: {e}")
        return False

    def save_codebook(self):
        with open(self.codebook_path, 'wb') as f:
            pickle.dump({
                'global_kmeans': self.global_kmeans,
                'local_kmeans': self.local_kmeans
            }, f)
        rospy.loginfo(f"Codebook saved to {self.codebook_path}")

    def extract_features(self, img):
        """Extract features using DINOv2"""
        # Convert OpenCV image to PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PILImage.fromarray(img)
        
        # Resize to maintain aspect ratio with shortest side 252
        original_w, original_h = img.size
        t_h, t_w = (252, int(252*(original_h/original_w))) if original_w > original_h \
                 else (int(252*(original_w/original_h)), 252)
        img = transforms.functional.resize(img, (t_h, t_w))
        
        # Calculate padding needs
        def _pad_spec(x): 
            pad = (14 - (x % 14)) % 14
            return (pad//2, pad - pad//2)
        
        w_pad = _pad_spec(img.width)
        h_pad = _pad_spec(img.height)
        img = transforms.functional.pad(img, (w_pad[0], h_pad[0], w_pad[1], h_pad[1]), fill=0)
        
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.forward_features(tensor)
        
        return {
            'global': features['x_norm_clstoken'].cpu().numpy().flatten(),
            'local': features['x_norm_patchtokens'].cpu().numpy().squeeze()
        }

    def _compute_vlad(self, features, kmeans):
        """Compute VLAD vector for features"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        cluster_ids = kmeans.predict(features)
        residuals = features - kmeans.cluster_centers_[cluster_ids]
        
        unique, counts = np.unique(cluster_ids, return_counts=True)
        weights = np.log(1 + counts[np.searchsorted(unique, cluster_ids)])
        
        vlad = np.zeros((self.num_clusters, features.shape[1]))
        np.add.at(vlad, cluster_ids, weights[:, None] * residuals)
        
        vlad = vlad.flatten()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        return vlad / (np.linalg.norm(vlad) + 1e-12)

    def generate_codebook(self):
        """Generate codebook from bag images"""
        global_features = []
        local_features = []
        
        try:
            with rosbag.Bag(self.bag_path, 'r') as bag:
                for _, msg, _ in bag.read_messages(topics=['/camera/image']):
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                        results = self.yolo.predict(source=cv_image, stream=True)
                        
                        for result in results:
                            for box in result.boxes.xyxy.cpu().numpy():
                                x1, y1, x2, y2 = map(int, box[:4])
                                crop = cv_image[y1:y2, x1:x2]
                                features = self.extract_features(crop)
                                
                                global_features.append(features['global'])
                                local_features.append(features['local'])
                                
                    except CvBridgeError as e:
                        rospy.logwarn(f"Skipping image: {e}")
                        continue
                        
        except Exception as e:
            rospy.logerr(f"Error processing bag file: {e}")
            raise

        # Train codebooks
        self.global_kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, 
                                           batch_size=1024,
                                           n_init=3).fit(np.array(global_features))
        
        self.local_kmeans = MiniBatchKMeans(n_clusters=self.num_clusters,
                                          batch_size=1024,
                                          n_init=3).fit(np.array(local_features))

    def build_vlad_database(self):
        """Build VLAD database from bag images"""
        try:
            with rosbag.Bag(self.bag_path, 'r') as bag:
                for _, msg, t in bag.read_messages(topics=['/camera/image']):
                    self.timestamps.append(t)
                    
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                        results = self.yolo.predict(source=cv_image, stream=True)
                        
                        for result in results:
                            for j, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                                x1, y1, x2, y2 = map(int, box[:4])
                                crop = cv_image[y1:y2, x1:x2]
                                features = self.extract_features(crop)
                                
                                global_vlad = self._compute_vlad(features['global'], self.global_kmeans)
                                local_vlad = self._compute_vlad(features['local'], self.local_kmeans)
                                combined = np.concatenate([global_vlad, local_vlad])
                                combined /= np.linalg.norm(combined) + 1e-12
                                
                                self.database.append(combined)
                                self.crop_indices.append((len(self.timestamps)-1, j))
                                
                    except CvBridgeError as e:
                        rospy.logwarn(f"Skipping image: {e}")
                        continue
                        
        except Exception as e:
            rospy.logerr(f"Error building VLAD database: {e}")
            raise

if __name__ == '__main__':
    try:
        Dinov2VLADProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 