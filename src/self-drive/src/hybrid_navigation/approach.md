# Hybrid Navigation System: Detailed Approach

This document provides a comprehensive explanation of the exploration and navigation approach implemented in our hybrid navigation system.

## System Architecture

Our hybrid navigation system combines visual place recognition, topological mapping, and hierarchical navigation to enable robust exploration and target-driven navigation in unknown environments. The system consists of three primary components:

1. **Visual Place Recognition (VPR)**: Responsible for feature extraction from camera images and matching with target locations
2. **Topological Mapper**: Builds and maintains a graph representation of the environment based on odometry data
3. **Hierarchical Navigator**: Controls the robot's movement using the topological map and visual recognition

![System Architecture](https://mermaid.ink/img/pako:eNp1kc1qwzAQhF9F7MkpuL_FJbcEcigthZxy8GFVNLEV5M1avBQ3wQ9UQim0r9RX6soKcdzGWMJodr558MoF1YhYYPM6TQdIe5d1QWCz9w1C5XQDhEo9QO26DLp2wKXzMUJudQfUxRkJQfXQgw4wR_MKfH45r1fZxQfF0HtKZzUGN6OPG_IZ3L-5YUNdSzpd8Hkv6KVrGmg41jXv7C6QWz5HoMYXMYr2TjMpRZIkRNbxpRKz2K0h84WsHlcgqm7l0dqKCO5ZFvzkf-Qk6UXp-GK_OB5zhQU1lPFIpVmUTFM5U8MpDRO9KtOKFnFU8TGP7ilNCB2v_Pfo6TAMLEwORYxJpGTKzl-MJIvThBey0rNcLLH4MpVWqFSc1ckUZVJmCu1X_AYtU5-T)

## Exploration Approach

### 1. Visual Feature Extraction

During exploration, the Visual Place Recognition (VPR) component continuously extracts features from camera images using a pre-trained CNN (ResNet18). These features form a high-dimensional descriptor for each image, capturing semantic information about the environment.

```python
def extract_features(self, img):
    # Preprocess image
    tensor = self.transform(img).unsqueeze(0)
    
    # Extract features using CNN
    with torch.no_grad():
        features = self.model(tensor)
    
    # Normalize features vector
    features = features.squeeze().cpu().numpy()
    features = features / np.linalg.norm(features)
    
    return features
```

These visual features are stored alongside corresponding odometry data, creating a database that associates visual appearances with physical locations in the environment.

### 2. Topological Map Construction

The Topological Mapper builds a graph representation of the environment where:
- **Nodes**: Represent distinct locations with stored position and orientation
- **Edges**: Connect nearby nodes, weighted by their physical distance

Our implementation creates a node for every odometry update, ensuring complete coverage of the robot's trajectory:

```python
# Exploration mode - create nodes for every odometry update
# Create a new node for this odometry reading
new_node_id = self.add_node(position, quat)
self.current_node_id = new_node_id

# Connect to nearby nodes
self.connect_nodes(new_node_id)
```

Node creation approach:
- A new node is created for every odometry update received
- Position and orientation are stored as 3D position vectors and quaternions
- Thresholds are set to extremely small values (0.001) to ensure all points are captured

Edge creation criteria:
- Maximum connection distance between nodes (`connection_threshold` = 0.3 meters)
- Edges are weighted by the Euclidean distance between nodes

The `connect_nodes` method establishes edges between nodes that are within the connection threshold:

```python
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
```

This approach creates a high-resolution map that captures every point in the robot's trajectory while maintaining a reasonable connectivity structure through the connection threshold.

### 3. Exploration Data Storage

During exploration, the system stores:
1. Visual descriptors associated with each node
2. Odometry data (position, orientation) for each node
3. Graph structure with nodes and weighted edges

This data is persisted to disk and can be reloaded for future navigation sessions:

```python
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
```

The map is automatically saved:
- When the ROS node is shut down
- When an explicit save request is received via the `/topological_map/save_request` topic

The Visual Place Recognition system also saves its database:

```python
def save_database(self):
    """Save the current database to disk"""
    if len(self.descriptors) > 0:
        self.build_database()
```

## Navigation Approach

Our hierarchical navigation system combines topological map-based navigation with visual recognition to create a robust solution for target-driven navigation. The system operates at multiple levels of abstraction, utilizing both the spatial structure of the environment and its visual appearance.

### 1. Navigation System Overview

The navigation process follows these key steps:

1. **Target Selection**: The user selects a target location by providing a target image
2. **Visual Localization**: The system localizes the target in the topological map using visual recognition
3. **Path Planning**: The system plans a path through the topological map from the current location to the target
4. **Waypoint Following**: The robot navigates through the planned path, moving from waypoint to waypoint
5. **Visual Verification**: Throughout navigation, the system continuously compares the current view with the target image
6. **Target Detection**: When a strong visual match is detected, the system considers the target reached

This hierarchical approach provides robustness through redundancy - even if visual recognition temporarily fails or odometry drift occurs, the system can recover and continue navigation.

### 2. Target Selection and Visual Localization

Navigation begins with loading a target image, which serves as the visual representation of the destination:

```python
def load_target_callback(self, req):
    """Load a target image and extract its features"""
    try:
        # Check if the image file exists
        if not os.path.exists(req.image_path):
            rospy.logerr(f"Target image not found: {req.image_path}")
            return False
        
        # Load the target image and extract features
        self.target_image = cv2.imread(req.image_path)
        if self.target_image is None:
            rospy.logerr(f"Failed to load target image: {req.image_path}")
            return False
        
        # Extract features from the target image
        self.target_descriptor = self.extract_features(self.target_image)
        
        # Reset best match tracking
        self.best_match_similarity = -1.0
        self.best_match_image = None
        self.best_match_path = None
        
        # Find the closest match in the database if available
        if self.database_built and self.tree is not None:
            # Query the tree for the closest descriptor
            dist, ind = self.tree.query(self.target_descriptor.reshape(1, -1), k=1)
            match_index = ind[0][0]
            self.match_index_pub.publish(Int32(match_index))
            rospy.loginfo(f"Target loaded: {req.image_path}. Closest database match: {match_index} (distance: {dist[0][0]:.4f})")
        else:
            rospy.loginfo(f"Target loaded: {req.image_path}. Database not built yet, no match found.")
        
        # Save the target image to the matches directory for reference
        target_copy_path = os.path.join(self.matches_dir, "target.jpg")
        cv2.imwrite(target_copy_path, self.target_image)
        
        return True
    except Exception as e:
        rospy.logerr(f"Error loading target: {e}")
        return False
```

The key steps in this process are:
1. Loading the target image from disk
2. Extracting a 512-dimensional feature vector using the CNN
3. Finding the closest match in the visual feature database using a BallTree (k-nearest neighbors)
4. Publishing the matched node ID for path planning

The system uses cosine similarity to determine matches between the current view and the target:

```python
# Compute similarity with current image features
similarity = np.dot(current_features, target_features)

# Determine if it's a match using a threshold
is_match = similarity >= self.match_threshold
```

### 3. Topological Path Planning

Once the target node is identified in the graph, the system plans a path from the current position to the target:

```python
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
```

The path planning process involves:
1. Determining the current position in the topological map
   - In navigation mode, this is the node where localization occurred
   - In exploration mode, this is the closest node to the current odometry position
2. Using NetworkX's implementation of Dijkstra's algorithm to find the shortest path
3. Weighing edges based on Euclidean distance between nodes
4. Publishing the complete path for visualization
5. Publishing the next waypoint for navigation

The next waypoint is published as a PoseStamped message containing the exact position and orientation of the node:

```python
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
```

With our high-resolution map (containing nodes at every odometry update), the path planning provides more detailed trajectories. The path is visualized in RViz using custom markers:

```python
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
```

### 4. Waypoint Navigation with PID Control

The Hierarchical Navigator implements sophisticated PID control to navigate sequentially through waypoints using odometry feedback:

```python
def navigate_to_waypoint(self):
    """Navigate to the current waypoint using PID control"""
    if self.current_pose is None or self.current_waypoint is None:
        return
    
    # Calculate distance and heading to waypoint
    target_pos = np.array([self.current_waypoint.position.x,
                          self.current_waypoint.position.y])
    current_pos = np.array([self.current_pose.position.x,
                           self.current_pose.position.y])
    
    # Linear distance error
    distance = np.linalg.norm(target_pos - current_pos)
    
    # If we're close enough to the waypoint
    if distance < self.distance_threshold:
        rospy.loginfo(f"Waypoint reached (distance: {distance:.2f}m)")
        self.current_waypoint = None
        return
    
    # Get current orientation as euler angles
    quaternion = (self.current_pose.orientation.x,
                 self.current_pose.orientation.y,
                 self.current_pose.orientation.z,
                 self.current_pose.orientation.w)
    _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
    
    # Calculate angle to target
    direction = target_pos - current_pos
    target_angle = np.arctan2(direction[1], direction[0])
    
    # Normalize angle difference to [-pi, pi]
    angle_diff = target_angle - yaw
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    # Calculate control using PID
    # Linear velocity control
    p_term_linear = self.kp_linear * distance
    self.integral_linear_error += distance
    i_term_linear = self.ki_linear * self.integral_linear_error
    d_term_linear = self.kd_linear * (distance - self.prev_linear_error)
    linear_velocity = p_term_linear + i_term_linear + d_term_linear
    
    # Cap linear velocity
    linear_velocity = min(self.linear_speed, max(-self.linear_speed, linear_velocity))
    
    # Angular velocity control
    p_term_angular = self.kp_angular * angle_diff
    self.integral_angular_error += angle_diff
    i_term_angular = self.ki_angular * self.integral_angular_error
    d_term_angular = self.kd_angular * (angle_diff - self.prev_angular_error)
    angular_velocity = p_term_angular + i_term_angular + d_term_angular
    
    # Cap angular velocity
    angular_velocity = min(self.angular_speed, max(-self.angular_speed, angular_velocity))
    
    # Update previous errors
    self.prev_linear_error = distance
    self.prev_angular_error = angle_diff
    
    # Create and publish velocity command
    cmd = Twist()
    
    # If the angle to target is too large, first rotate in place
    if abs(angle_diff) > self.angle_threshold:
        cmd.linear.x = 0.0
        cmd.angular.z = angular_velocity
        self.status_pub.publish(String("Rotating to target"))
    else:
        # Otherwise move forward while adjusting heading
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity
        self.status_pub.publish(String("Moving to target"))
    
    self.cmd_vel_pub.publish(cmd)
```

The waypoint navigation strategy involves:

1. **Distance Calculation**: Determining the 2D Euclidean distance to the waypoint
2. **Orientation Calculation**: Computing the heading difference to the waypoint
3. **PID Control for Linear Velocity**:
   - P-term: Proportional to the distance from the waypoint (kp_linear = 0.5)
   - I-term: Integrates distance error over time (ki_linear = 0.0)
   - D-term: Responds to rate of change of distance (kd_linear = 0.0)
4. **PID Control for Angular Velocity**:
   - P-term: Proportional to the angle difference (kp_angular = 1.0)
   - I-term: Integrates angular error over time (ki_angular = 0.0)
   - D-term: Responds to rate of change of angle (kd_angular = 0.1)
5. **Control Strategy Switching**:
   - When angle error is large (> angle_threshold), robot rotates in place
   - When angle error is small, robot moves forward while adjusting heading

The controller is tuned to prioritize accurate heading alignment before moving forward, resulting in more precise navigation. Parameters like linear_speed (0.2 m/s) and angular_speed (0.5 rad/s) limit the maximum velocities for safety.

When a waypoint is reached (distance < distance_threshold), the system requests the next waypoint in the path until the entire route is traversed.

### 5. Visual Target Recognition

While navigating through the topological map, the system continuously monitors the visual similarity between the current camera view and the target image:

```python
def vpr_match_callback(self, msg):
    """Handle visual place recognition match updates"""
    self.vpr_match = msg.data
    
    # If we have a match, we reached the target
    if self.vpr_match:
        self.target_reached = True
        self.stop_robot()
        self.target_reached_pub.publish(Bool(True))
        rospy.loginfo(f"Target reached based on visual recognition (similarity: {self.vpr_similarity})")
```

This visual verification layer serves several purposes:
1. **Target Confirmation**: Verifies when the robot has actually reached the visual target
2. **Early Stopping**: Allows stopping navigation as soon as the target is visually recognized, even if the waypoint isn't reached
3. **Robustness to Map Errors**: Even if the topological map contains errors, the system can still find the target

The visual place recognition similarity score is continuously published:

```python
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
```

### 6. Best Match Tracking and Visualization

To assist with monitoring navigation progress, the system tracks and visualizes the best matching views to the target:

```python
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
```

This visualization system:
1. Keeps track of the best visual match seen so far
2. Saves images when a new best match is found
3. Allows the user to monitor navigation progress through a separate visualization node

### 7. Recovery and Error Handling

The navigation system includes several recovery mechanisms to handle errors and edge cases:

1. **Localization Recovery**: If localization is lost, the system can re-localize using visual recognition
2. **Path Replanning**: If the robot deviates from the path, a new path is planned from the closest node
3. **Target Reacquisition**: Visual recognition can detect the target even if path following fails
4. **Obstacle Handling**: The robot stops and attempts to replan if a waypoint becomes unreachable
5. **Emergency Stop**: A safety measure that stops the robot when unexpected conditions are detected

```python
def stop_robot(self):
    """Stop the robot by publishing zero velocity"""
    cmd = Twist()
    cmd.linear.x = 0.0
    cmd.angular.z = 0.0
    
    # Publish several times to ensure the robot stops
    for _ in range(3):
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(0.1)
```

## Key Algorithms

### 1. Visual Feature Extraction

- **Model**: ResNet18 pre-trained on ImageNet, with final FC layer removed
- **Input**: 224×224 RGB images, normalized with ImageNet stats
- **Output**: 512-dimensional feature vector, L2-normalized

Feature extraction code:
```python
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
```

### 2. Visual Similarity Calculation

- **Method**: Cosine similarity between feature vectors
- **Formula**: `similarity = np.dot(features, self.target_descriptor)`
- **Threshold**: Configurable via `best_match_threshold` parameter (default: 0.75)

```python
# Compute similarity with target descriptor
similarity = np.dot(features, self.target_descriptor)
self.similarity_pub.publish(Float32(similarity))

# Publish match based on threshold
is_match = similarity >= self.best_match_threshold
self.match_pub.publish(Bool(is_match))
```

### 3. Path Planning

- **Algorithm**: Dijkstra's shortest path on weighted graph
- **Weights**: Euclidean distance between node positions
- **Implementation**: Uses NetworkX library for graph operations

### 4. PID Controller for Navigation

The PID controller adjusts linear and angular velocities based on distance and angle errors:

```python
# Linear velocity control
p_term_linear = self.kp_linear * distance
self.integral_linear_error += distance
i_term_linear = self.ki_linear * self.integral_linear_error
d_term_linear = self.kd_linear * (distance - self.prev_linear_error)
linear_velocity = p_term_linear + i_term_linear + d_term_linear

# Angular velocity control
p_term_angular = self.kp_angular * angle_diff
self.integral_angular_error += angle_diff
i_term_angular = self.ki_angular * self.integral_angular_error
d_term_angular = self.kd_angular * (angle_diff - self.prev_angular_error)
angular_velocity = p_term_angular + i_term_angular + d_term_angular
```

Parameters:
- **Linear Control**: kp_linear=0.5, ki_linear=0.0, kd_linear=0.0 (default values)
- **Angular Control**: kp_angular=1.0, ki_angular=0.0, kd_angular=0.1 (default values)
- **Speed Limits**: linear_speed=0.2 m/s, angular_speed=0.5 rad/s (configurable)

## Advantages of the Hybrid Approach

1. **Robustness**: By combining visual recognition with topological mapping, the system can handle visual ambiguities and occlusions
2. **High-Resolution Mapping**: By storing all odometry points as nodes, we capture the complete trajectory of the robot
3. **Detailed Path Planning**: More nodes enable more precise navigation paths
4. **Adaptability**: The system can navigate through modified environments as long as key visual landmarks remain recognizable
5. **Loop Closure**: Visual recognition naturally handles loop closures in the topological map
6. **No Global Localization**: No need for precise global localization as navigation is relative to the topological map

## Limitations and Future Work

1. **Map Size**: Storing every odometry point increases the map size compared to keyframe-based approaches
2. **Computational Overhead**: More nodes require more processing during path planning
3. **Visual Changes**: Performance degrades with significant lighting or seasonal changes
4. **Dynamic Environments**: The current implementation assumes a static environment
5. **Viewpoint Sensitivity**: Visual recognition is somewhat sensitive to viewpoint changes

Future improvements could include:
- Adding map compression techniques to reduce redundancy while maintaining detail
- Implementing node pruning to optimize the map after creation
- Adding semantic information to enhance path planning
- Implementing dynamic map updates during navigation
- Developing more viewpoint-invariant visual descriptors
- Integrating obstacle avoidance for safer navigation
- Implementing map merging capabilities for collaborative exploration
- Adding active loop closure detection for improved map consistency 