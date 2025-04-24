#!/usr/bin/env python3

import rospy
import numpy as np
import pickle
import os
import networkx as nx
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Int32, Int32MultiArray
from hybrid_navigation.srv import LoadTarget
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class TopologicalMapper:
    def __init__(self):
        rospy.init_node('topological_mapper')
        
        # Parameters
        self.map_path = rospy.get_param('~map_path', 'topological_map.pkl')
        # Set a very small distance threshold to ensure every update creates a node
        self.keyframe_distance = rospy.get_param('~keyframe_distance', 0.001)  # meters (very small)
        self.keyframe_angle = rospy.get_param('~keyframe_angle', 0.001)  # radians (very small)
        self.connection_threshold = rospy.get_param('~connection_threshold', 0.3)  # meters
        self.navigation_mode = rospy.get_param('~navigation_mode', False)  # Set to True when in navigation mode
        
        # Initialize the graph
        self.graph = nx.Graph()
        self.node_count = 0
        self.last_pose = None
        self.current_pose = None
        self.target_node = None
        self.localized = False  # Whether we've localized in the map during navigation
        self.current_node_id = None  # Current node ID in the graph
        
        # Position and orientation history (for visualization)
        self.positions = []
        self.orientations = []
        
        # Publishers for visualization
        self.markers_pub = rospy.Publisher('/topological_map/visualization', MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher('/topological_map/path', Int32MultiArray, queue_size=10)
        self.current_node_pub = rospy.Publisher('/topological_map/current_node', Int32, queue_size=10)
        self.next_waypoint_pub = rospy.Publisher('/topological_map/next_waypoint', PoseStamped, queue_size=10)
        
        # Subscribers
        # Reduce the queue size to ensure we process all odometry updates
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        self.target_sub = rospy.Subscriber('/vpr/match_index', Int32, self.target_callback)
        
        # Add a subscriber to VPR match for localization
        self.vpr_localization_sub = rospy.Subscriber('/vpr/match_index', Int32, self.localization_callback)
        
        # Add a subscriber to handle save requests
        self.save_request_sub = rospy.Subscriber('/topological_map/save_request', Int32, self.save_request_callback)
        
        # Signal handler for saving the map on shutdown
        rospy.on_shutdown(self.save_topological_map)
        
        # Load existing map if available
        if os.path.exists(self.map_path):
            self.load_topological_map()
            
            if self.navigation_mode:
                rospy.loginfo("Starting in NAVIGATION mode - waiting for visual localization...")
            else:
                rospy.loginfo("Starting in EXPLORATION mode - will create new nodes.")
        else:
            rospy.loginfo("No existing topological map found. Creating a new one.")
            # Always in exploration mode if no map exists
            self.navigation_mode = False
            self.localized = True  # No need to localize in exploration
        
        rospy.loginfo("Topological Mapper initialized - Set to save ALL nodes")
    
    def localization_callback(self, msg):
        """Handle visual place recognition match for localization"""
        if self.navigation_mode and not self.localized:
            match_node_id = msg.data
            
            # Check if the node exists in our graph
            if match_node_id in self.graph.nodes:
                self.current_node_id = match_node_id
                self.localized = True
                self.current_node_pub.publish(Int32(match_node_id))
                rospy.loginfo(f"Successfully localized at node {match_node_id} using visual recognition")
    
    def odom_callback(self, msg):
        """Process odometry updates and update the graph"""
        try:
            # Extract position and orientation
            pose = msg.pose.pose
            
            # TurtleBot3 is a 2D robot - use only x,y for position
            # Z coordinate is fixed at 0 for planar movement
            position = np.array([pose.position.x, pose.position.y, 0.0])
            
            # For 2D movement, only z and w components of the quaternion are used
            # The x and y components are typically 0
            quat = np.array([0.0, 0.0, pose.orientation.z, pose.orientation.w])
            
            # Normalize the quaternion to ensure it's valid
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 0:
                quat = quat / quat_norm
            
            # Save current pose
            self.current_pose = (position, quat)
            
            # Track all positions for visualization
            self.positions.append(position)
            self.orientations.append(quat)
            
            # In navigation mode, we don't create new nodes unless needed
            if self.navigation_mode:
                if not self.localized:
                    # We haven't localized yet, so don't create nodes
                    rospy.logdebug("Waiting for visual localization before creating nodes...")
                    return
                
                # We're localized - find the closest existing node to our current position
                closest_node, min_distance = self.find_closest_node(position)
                
                # If we're close enough to an existing node, just use that
                if min_distance <= self.connection_threshold:
                    self.current_node_id = closest_node
                    self.current_node_pub.publish(Int32(closest_node))
                    return
            
            # Exploration mode - create nodes for every odometry update
            
            # If this is the first reading, just record it
            if self.last_pose is None:
                self.last_pose = self.current_pose
                
                # Add first node to the graph
                node_id = self.add_node(position, quat)
                self.current_node_id = node_id
                return
            
            # Create a new node for this odometry reading
            new_node_id = self.add_node(position, quat)
            self.current_node_id = new_node_id
            
            # Connect to nearby nodes
            self.connect_nodes(new_node_id)
            
            # Update last pose
            self.last_pose = self.current_pose
            
            # Update visualization
            self.publish_visualization()
            
            # Publish current node
            self.current_node_pub.publish(Int32(new_node_id))
            
        except Exception as e:
            rospy.logerr(f"Error in odom_callback: {e}")
    
    def find_closest_node(self, position):
        """Find the closest node to a given position"""
        closest_node = None
        min_distance = float('inf')
        
        for node in self.graph.nodes:
            if 'position' not in self.graph.nodes[node]:
                continue
                
            node_position = self.graph.nodes[node]['position']
            # Only consider x,y for distance (2D)
            distance = np.linalg.norm(position[:2] - node_position[:2])
            
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        return closest_node, min_distance
    
    def add_node(self, position, orientation):
        """Add a new node to the topological map at the current position"""
        try:
            # Create a new node with the current position and orientation
            node_id = self.node_count
            
            # Store position and orientation as numpy arrays to ensure consistency
            # Check if position is already a numpy array
            if isinstance(position, np.ndarray):
                pos_np = position
            else:
                # Handle case when position is an object with x, y, z attributes
                pos_np = np.array([position.x, position.y, position.z])
            
            # Check if orientation is already a numpy array
            if isinstance(orientation, np.ndarray):
                quat_np = orientation
            else:
                # Handle case when orientation is an object with x, y, z, w attributes
                quat_np = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
            
            # Normalize the quaternion
            quat_norm = np.linalg.norm(quat_np)
            if quat_norm > 0:
                quat_np = quat_np / quat_norm
            
            # Add the node to the graph with the attributes
            self.graph.add_node(
                node_id, 
                position=pos_np,
                orientation=quat_np
            )
            
            # Connect to nearby nodes (if within connection_threshold)
            for existing_node in self.graph.nodes:
                if existing_node == node_id:
                    continue
                
                existing_pos = self.graph.nodes[existing_node]['position']
                distance = np.linalg.norm(pos_np[:2] - existing_pos[:2])  # 2D distance
                
                if distance <= self.connection_threshold:
                    # Add edge with distance as weight
                    self.graph.add_edge(node_id, existing_node, weight=distance)
            
            # Increment node counter
            self.node_count += 1
            
            # Publish updated visualization
            self.publish_visualization()
            
            # Log information
            rospy.loginfo(f"Added node {node_id} at position {pos_np}")
            return node_id
            
        except Exception as e:
            rospy.logerr(f"Error adding node: {e}")
            return None
    
    def connect_nodes(self, node_id):
        """Connect a node to nearby nodes in the graph"""
        try:
            # Check if the node exists and has position attribute
            if node_id not in self.graph.nodes or 'position' not in self.graph.nodes[node_id]:
                rospy.logwarn(f"Node {node_id} doesn't exist or doesn't have position attribute")
                return
                
            node_position = self.graph.nodes[node_id]['position']
            
            # Check all other nodes for potential connections
            for other_id in list(self.graph.nodes):
                if other_id == node_id:
                    continue
                    
                # Skip nodes that don't have position attribute
                if 'position' not in self.graph.nodes[other_id]:
                    rospy.logwarn(f"Node {other_id} doesn't have position attribute")
                    continue
                    
                other_position = self.graph.nodes[other_id]['position']
                
                # Calculate 2D distance (only x and y coordinates)
                distance = np.linalg.norm(node_position[:2] - other_position[:2])
                
                # Connect nodes if they are close enough
                if distance < self.connection_threshold:
                    self.graph.add_edge(node_id, other_id, weight=distance)
                    rospy.logdebug(f"Connected nodes {node_id} and {other_id} with distance {distance}")
        except Exception as e:
            rospy.logerr(f"Error connecting nodes: {e}")
    
    def quaternion_distance(self, q1, q2):
        """Calculate angular distance between two quaternions"""
        try:
            # For 2D rotation (only around z-axis), we can simplify this
            # The dot product between quaternions
            dot_product = np.abs(np.sum(q1 * q2))
            dot_product = min(1.0, max(-1.0, dot_product))  # Clamp to [-1, 1]
            
            # Convert to angle
            angle = 2 * np.arccos(dot_product)
            return angle
        except Exception as e:
            rospy.logerr(f"Error calculating quaternion distance: {e}")
            return 0.0
    
    def target_callback(self, msg):
        """Process target place recognition and plan a path"""
        try:
            # Check if we have a valid place recognition
            node_id = msg.data
            
            # Check if the node exists in the graph
            if node_id not in self.graph.nodes:
                rospy.logwarn(f"Node {node_id} not found in graph")
                return
                
            # If not localized yet or no current node ID is set, we can't plan a path
            if self.navigation_mode and (not self.localized or self.current_node_id is None):
                rospy.logwarn("Not localized yet. Cannot plan path to target.")
                return
                
            # Use current_node_id as the start node if we're localized
            if self.navigation_mode and self.localized and self.current_node_id is not None:
                closest_node = self.current_node_id
            else:
                # Find the closest current node using odometry
                if self.current_pose is None:
                    rospy.logwarn("No current pose available")
                    return
                    
                position, _ = self.current_pose
                closest_node, _ = self.find_closest_node(position)
                
                if closest_node is None:
                    rospy.logwarn("No closest node found")
                    return
            
            # Plan a path using Dijkstra's algorithm
            try:
                path = nx.shortest_path(self.graph, closest_node, node_id, weight='weight')
                rospy.loginfo(f"Found path from {closest_node} to {node_id}: {path}")
                
                # Publish the path
                self.path_pub.publish(Int32MultiArray(data=path))
                
                # If path has at least one waypoint, publish the next one
                if len(path) > 1:
                    next_node = path[1]  # The next node to visit
                    self.publish_next_waypoint(next_node)
            except nx.NetworkXNoPath:
                rospy.logwarn(f"No path found from {closest_node} to {node_id}")
            except Exception as e:
                rospy.logerr(f"Error planning path: {e}")
        except Exception as e:
            rospy.logerr(f"Error in target_callback: {e}")
    
    def publish_next_waypoint(self, node_id):
        """Publish the next waypoint to navigate to"""
        try:
            # Check if the node exists
            if node_id not in self.graph.nodes:
                rospy.logwarn(f"Node {node_id} not found in graph")
                return
                
            # Get node position and orientation
            node_data = self.graph.nodes[node_id]
            
            # Check if node has required attributes
            if 'position' not in node_data or 'orientation' not in node_data:
                rospy.logwarn(f"Node {node_id} missing required attributes")
                return
                
            # Create PoseStamped message
            waypoint = PoseStamped()
            waypoint.header.stamp = rospy.Time.now()
            waypoint.header.frame_id = "map"  # Use map frame
            
            # Set position (x, y, z)
            position = node_data['position']
            waypoint.pose.position.x = position[0]
            waypoint.pose.position.y = position[1]
            waypoint.pose.position.z = position[2]  # Should be 0 for 2D
            
            # Set orientation (quaternion)
            orientation = node_data['orientation']
            waypoint.pose.orientation.x = orientation[0]  # Should be 0 for 2D
            waypoint.pose.orientation.y = orientation[1]  # Should be 0 for 2D
            waypoint.pose.orientation.z = orientation[2]
            waypoint.pose.orientation.w = orientation[3]
            
            # Publish the waypoint
            self.next_waypoint_pub.publish(waypoint)
            rospy.loginfo(f"Published next waypoint for node {node_id}")
        except Exception as e:
            rospy.logerr(f"Error publishing waypoint: {e}")
    
    def publish_visualization(self):
        """Publish visualization markers for the topological map"""
        try:
            marker_array = MarkerArray()
            
            # Node markers
            node_marker = Marker()
            node_marker.header.frame_id = "odom"
            node_marker.header.stamp = rospy.Time.now()
            node_marker.ns = "nodes"
            node_marker.id = 0
            node_marker.type = Marker.SPHERE_LIST
            node_marker.action = Marker.ADD
            node_marker.scale.x = 0.2
            node_marker.scale.y = 0.2
            node_marker.scale.z = 0.2
            node_marker.color.r = 0.0
            node_marker.color.g = 1.0
            node_marker.color.b = 0.0
            node_marker.color.a = 1.0
            
            # Edge markers
            edge_marker = Marker()
            edge_marker.header.frame_id = "odom"
            edge_marker.header.stamp = rospy.Time.now()
            edge_marker.ns = "edges"
            edge_marker.id = 1
            edge_marker.type = Marker.LINE_LIST
            edge_marker.action = Marker.ADD
            edge_marker.scale.x = 0.05
            edge_marker.color.r = 1.0
            edge_marker.color.g = 1.0
            edge_marker.color.b = 1.0
            edge_marker.color.a = 0.5
            
            # Add nodes
            for node_id in self.graph.nodes:
                if 'position' not in self.graph.nodes[node_id]:
                    rospy.logwarn(f"Node {node_id} doesn't have position attribute, skipping visualization")
                    continue
                    
                position = self.graph.nodes[node_id]['position']
                p = Point()
                p.x = position[0]
                p.y = position[1]
                p.z = position[2]
                node_marker.points.append(p)
            
            # Add edges
            for u, v in self.graph.edges:
                if 'position' not in self.graph.nodes[u] or 'position' not in self.graph.nodes[v]:
                    rospy.logwarn(f"Edge {u}-{v} connects nodes without position attributes, skipping visualization")
                    continue
                    
                pos_u = self.graph.nodes[u]['position']
                pos_v = self.graph.nodes[v]['position']
                
                p1 = Point()
                p1.x = pos_u[0]
                p1.y = pos_u[1]
                p1.z = pos_u[2]
                
                p2 = Point()
                p2.x = pos_v[0]
                p2.y = pos_v[1]
                p2.z = pos_v[2]
                
                edge_marker.points.append(p1)
                edge_marker.points.append(p2)
            
            marker_array.markers.append(node_marker)
            marker_array.markers.append(edge_marker)
            
            self.markers_pub.publish(marker_array)
        except Exception as e:
            rospy.logerr(f"Error publishing visualization: {e}")
    
    def publish_path_visualization(self, path):
        """Publish visualization for the path to target"""
        try:
            marker_array = MarkerArray()
            
            # Path marker
            path_marker = Marker()
            path_marker.header.frame_id = "odom"
            path_marker.header.stamp = rospy.Time.now()
            path_marker.ns = "path"
            path_marker.id = 0
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.1
            path_marker.color.r = 1.0
            path_marker.color.g = 0.0
            path_marker.color.b = 0.0
            path_marker.color.a = 1.0
            
            # Add path points
            for node_id in path:
                if node_id not in self.graph.nodes or 'position' not in self.graph.nodes[node_id]:
                    rospy.logwarn(f"Node {node_id} in path is missing position attribute, skipping")
                    continue
                    
                position = self.graph.nodes[node_id]['position']
                p = Point()
                p.x = position[0]
                p.y = position[1]
                p.z = position[2]
                path_marker.points.append(p)
            
            marker_array.markers.append(path_marker)
            self.path_pub.publish(marker_array)
        except Exception as e:
            rospy.logerr(f"Error publishing path visualization: {e}")
    
    def save_request_callback(self, msg):
        """Handle request to save the map"""
        try:
            rospy.loginfo("Received request to save topological map")
            success = self.save_topological_map()
            if success:
                rospy.loginfo(f"Topological map saved successfully with {len(self.graph.nodes)} nodes")
            else:
                rospy.logwarn("Failed to save topological map")
        except Exception as e:
            rospy.logerr(f"Error handling save request: {e}")
            
    def save_topological_map(self):
        """Save the topological map to a file"""
        try:
            # Ensure graph nodes have all necessary attributes
            nodes_to_check = list(self.graph.nodes)
            for node_id in nodes_to_check:
                node_data = self.graph.nodes[node_id]
                
                # Check if node has required attributes
                if 'position' not in node_data:
                    rospy.logwarn(f"Node {node_id} is missing position. Adding zero position.")
                    self.graph.nodes[node_id]['position'] = np.zeros(3)
                    
                if 'orientation' not in node_data:
                    rospy.logwarn(f"Node {node_id} is missing orientation. Adding zero quaternion.")
                    self.graph.nodes[node_id]['orientation'] = np.array([0.0, 0.0, 0.0, 1.0])
                    
                # Ensure position is a numpy array with 3 elements
                if not isinstance(node_data['position'], np.ndarray) or len(node_data['position']) != 3:
                    rospy.logwarn(f"Node {node_id} has invalid position format. Fixing it.")
                    pos = node_data['position']
                    if isinstance(pos, (list, tuple)) and len(pos) == 3:
                        self.graph.nodes[node_id]['position'] = np.array(pos)
                    else:
                        self.graph.nodes[node_id]['position'] = np.zeros(3)
                        
                # Ensure orientation is a numpy array with 4 elements (quaternion)
                if not isinstance(node_data['orientation'], np.ndarray) or len(node_data['orientation']) != 4:
                    rospy.logwarn(f"Node {node_id} has invalid orientation format. Fixing it.")
                    quat = node_data['orientation']
                    if isinstance(quat, (list, tuple)) and len(quat) == 4:
                        self.graph.nodes[node_id]['orientation'] = np.array(quat)
                    else:
                        self.graph.nodes[node_id]['orientation'] = np.array([0.0, 0.0, 0.0, 1.0])
                        
                # Normalize the quaternion to ensure it's valid
                quat = node_data['orientation']
                quat_norm = np.linalg.norm(quat)
                if quat_norm > 0:
                    self.graph.nodes[node_id]['orientation'] = quat / quat_norm
            
            # Create directory if it doesn't exist
            map_dir = os.path.dirname(self.map_path)
            if map_dir and not os.path.exists(map_dir):
                os.makedirs(map_dir)
            
            # Save the graph to a pickle file
            with open(self.map_path, 'wb') as f:
                pickle.dump(self.graph, f)
                
            rospy.loginfo(f"Saved topological map with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges to {self.map_path}")
            return True
        except Exception as e:
            rospy.logerr(f"Error saving topological map: {e}")
            return False

    def load_topological_map(self):
        """Load the topological map from a file"""
        try:
            if not os.path.exists(self.map_path):
                rospy.logwarn("No topological map found. Starting with empty map.")
                return
                
            # Load the graph from the pickle file
            with open(self.map_path, 'rb') as f:
                loaded_graph = pickle.load(f)
                
            # Validate and clean the loaded graph
            nodes_to_remove = []
            for node_id in loaded_graph.nodes:
                node_data = loaded_graph.nodes[node_id]
                
                # Check if node has required attributes
                if 'position' not in node_data or 'orientation' not in node_data:
                    nodes_to_remove.append(node_id)
                    rospy.logwarn(f"Node {node_id} is missing required attributes. Removing it from the graph.")
                    continue
                    
                # Ensure position is a numpy array with 3 elements
                if not isinstance(node_data['position'], np.ndarray) or len(node_data['position']) != 3:
                    nodes_to_remove.append(node_id)
                    rospy.logwarn(f"Node {node_id} has invalid position format. Removing it from the graph.")
                    continue
                    
                # Ensure orientation is a numpy array with 4 elements (quaternion)
                if not isinstance(node_data['orientation'], np.ndarray) or len(node_data['orientation']) != 4:
                    nodes_to_remove.append(node_id)
                    rospy.logwarn(f"Node {node_id} has invalid orientation format. Removing it from the graph.")
                    continue
                    
                # Normalize the quaternion to ensure it's valid
                quat = node_data['orientation']
                quat_norm = np.linalg.norm(quat)
                if quat_norm > 0:
                    node_data['orientation'] = quat / quat_norm
            
            # Remove invalid nodes
            for node_id in nodes_to_remove:
                loaded_graph.remove_node(node_id)
                
            if nodes_to_remove:
                rospy.logwarn("Some nodes were removed from the graph due to missing attributes.")
                
            # Update our graph
            self.graph = loaded_graph
            
            # Update node count to be one more than the highest node ID
            if self.graph.nodes:
                max_node_id = max([int(node_id) for node_id in self.graph.nodes if isinstance(node_id, (int, str)) and str(node_id).isdigit()])
                self.node_count = max_node_id + 1
            else:
                self.node_count = 0
                
            rospy.loginfo(f"Loaded topological map with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        except Exception as e:
            rospy.logerr(f"Error loading topological map: {e}")
            rospy.logwarn("Starting with empty map due to loading error.")
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Regular processing
            rate.sleep()
        
        # Save map on shutdown
        self.save_topological_map()

if __name__ == '__main__':
    try:
        mapper = TopologicalMapper()
        mapper.run()
    except rospy.ROSInterruptException:
        pass 