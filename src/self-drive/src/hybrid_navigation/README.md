# Hybrid Navigation System

This package implements a hybrid navigation approach combining visual place recognition with odometry-based path optimization for TurtleBot3.

## Overview

The hybrid navigation system uses a topological map-based approach where:
1. **Visual Keyframes**: CNN-based descriptors (ResNet18) establish nodes in a graph
2. **Odometry Edges**: Connect graph nodes with relative odometry displacements
3. **Path Planning**: Find shortest path through graph using Dijkstra's algorithm
4. **Visual Re-identification**: Recognizes target location using visual similarity

## Architecture

The system has three main components:

1. **Visual Place Recognition (VPR)**: Uses pre-trained CNN to extract features from camera images and compares them with a target image.
2. **Topological Mapper**: Builds and maintains a graph representation of the environment.
3. **Hierarchical Navigator**: Controls robot movement based on waypoints from the topological map and VPR.

## Usage

### Exploration Phase

First, explore the environment to build the topological map:

```bash
roslaunch hybrid_navigation mapping.launch
```

This will:
- Record a rosbag with `/odom`, `/camera/image`, and `/cmd_vel` topics
- Allow you to teleoperate the robot
- Build a topological map and visual place recognition database

### Navigation Phase

To navigate to a previously visited location:

```bash
roslaunch hybrid_navigation navigation.launch target_image_path:=/path/to/target/image.jpg
```

To view best match
```bash
roslaunch hybrid_navigation view_match.launch
```

This will:
- Load the topological map and visual place recognition database
- Find the target location in the map
- Navigate to the target using topological path following

## Parameters

### Visual Place Recognition
- `model_name`: CNN model to use (default: "resnet18")
- `database_path`: Path to save/load the VPR database
- `similarity_threshold`: Threshold for target recognition (default: 0.85)

### Topological Mapper
- `map_path`: Path to save/load the topological map
- `keyframe_distance`: Minimum distance for adding a new node (default: 0.5m)
- `keyframe_angle`: Minimum angle change for adding a new node (default: 0.5rad)
- `connection_distance`: Maximum distance to connect nodes (default: 2.0m)

### Hierarchical Navigator
- `linear_speed`: Maximum linear speed (default: 0.2m/s)
- `angular_speed`: Maximum angular speed (default: 0.5rad/s)
- `distance_threshold`: Waypoint reaching threshold (default: 0.3m)
- `angle_threshold`: Angle alignment threshold (default: 0.1rad)

## Requirements

- ROS Noetic
- Python 3
- PyTorch
- OpenCV
- NetworkX
- scikit-learn

## References

This implementation is based on the hybrid approach described in the solution document. 