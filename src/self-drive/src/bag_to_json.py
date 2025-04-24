#!/usr/bin/env python3

import rosbag
import json
import argparse
import os
import yaml
import base64
from genpy import Time, Duration

# Add these imports for image processing
import cv2
from cv_bridge import CvBridge

def create_dir_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def ros_message_to_dict(msg, topic=None, timestamp=None, image_dir=None, image_count=None):
    """
    Convert a ROS message object into a dictionary.
    For image messages, saves the image to disk and returns path reference.
    """
    # Handle Time/Duration objects
    if isinstance(msg, Time) or isinstance(msg, Duration):
        return {'secs': msg.secs, 'nsecs': msg.nsecs}
    
    # Handle image messages for known image topics
    if image_dir is not None and topic is not None and timestamp is not None and image_count is not None:
        if topic == '/camera/image' or 'image' in topic:
            try:
                # Create a directory for this topic if it doesn't exist
                topic_dir = os.path.join(image_dir, topic.replace('/', '_').lstrip('_'))
                create_dir_if_not_exists(topic_dir)
                
                # Convert to image and save
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                
                # Create unique filename with timestamp and message number
                filename = f"{image_count:08d}.jpg"
                filepath = os.path.join(topic_dir, filename)
                
                # Save the image
                cv2.imwrite(filepath, cv_image)
                
                # Return the path instead of the image data
                return {"image_path": filepath}
            except Exception as e:
                print(f"Error processing image in topic {topic}: {e}")
                # If there's an error, continue with default processing
    
    # Handle ROS messages with slots
    if hasattr(msg, '__slots__'):
        data = {}
        for slot in msg.__slots__:
            # Recursively process each slot
            data[slot] = ros_message_to_dict(getattr(msg, slot), topic, timestamp, image_dir, image_count)
        return data
    
    # Handle lists/tuples recursively
    elif isinstance(msg, list) or isinstance(msg, tuple):
        return [ros_message_to_dict(item, topic, timestamp, image_dir, image_count) for item in msg]
    
    # Handle numpy arrays
    elif type(msg).__module__ == 'numpy' and type(msg).__name__ == 'ndarray':
        return msg.tolist()
    
    # Handle bytes by base64 encoding
    elif isinstance(msg, bytes):
        # Limit size of binary data we encode to avoid huge JSON files
        if len(msg) > 1024:  # If larger than 1KB
            return {"binary_data_size": len(msg), "note": "Large binary data omitted from JSON"}
        else:
            # For small binary data, encode as base64
            return {"binary_data": base64.b64encode(msg).decode('ascii')}
    
    # Handle other primitive types
    else:
        return msg

def bag_to_json(bag_file, json_file):
    """Reads a ROS bag file and writes its contents to a JSON file."""
    print(f"Reading bag file: {bag_file}")
    if not os.path.exists(bag_file):
        print(f"Error: Bag file not found at {bag_file}")
        return

    # Setup directory for saving images
    bag_dir = os.path.dirname(bag_file)
    bag_name = os.path.splitext(os.path.basename(bag_file))[0]
    image_dir = os.path.join(bag_dir, f"{bag_name}_images")
    create_dir_if_not_exists(image_dir)
    
    bag_data = {}
    message_count = 0
    topic_message_counts = {}

    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            # Get info about topics
            info_dict = yaml.safe_load(bag._get_yaml_info())
            print(f"Bag contains {len(info_dict['topics'])} topics")
            
            for topic, msg, t in bag.read_messages():
                # Initialize topic counter if needed
                if topic not in topic_message_counts:
                    topic_message_counts[topic] = 0
                    bag_data[topic] = []
                
                # Get message count for this topic
                topic_count = topic_message_counts[topic]
                topic_message_counts[topic] += 1
                
                # Convert message and timestamp to dictionaries
                msg_dict = ros_message_to_dict(msg, topic, t, image_dir, topic_count)
                timestamp_dict = {'secs': t.secs, 'nsecs': t.nsecs}

                bag_data[topic].append({
                    'timestamp': timestamp_dict,
                    'message': msg_dict
                })
                
                message_count += 1
                if message_count % 100 == 0:
                    print(f"Processed {message_count} messages...")

    except rosbag.bag.ROSBagException as e:
        print(f"Error reading bag file: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print(f"Finished processing {message_count} messages.")
    print(f"Writing data to JSON file: {json_file}")
    print(f"Images saved to: {image_dir}")

    try:
        with open(json_file, 'w') as f:
            json.dump(bag_data, f, indent=4)
        print("Successfully wrote JSON output.")
    except IOError as e:
        print(f"Error writing JSON file: {e}")
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}")
        print("This might happen with complex or non-standard message types.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a ROS bag file to a JSON file.')
    # Make bag_file optional and add a default value
    # Note: This default path is absolute based on the user's previous request.
    # Consider making it relative or removing the default if the project moves.
    parser.add_argument('bag_file', nargs='?', 
                        default='/home/ai4ce/team3-deploy/catkin_s25/bags/custom_name_2025-03-14-13-26-00.bag',
                        help='Path to the input ROS bag file. Defaults to a specific bag if not provided.')
    parser.add_argument('-o', '--output', help='Path to the output JSON file. Defaults to <bag_file_name>.json in the bag file\'s directory.')

    args = parser.parse_args()

    # Determine output file path
    # This logic already defaults to the input bag file's directory
    if args.output:
        json_file_path = args.output
    else:
        # Get the directory of the input bag file
        bag_dir = os.path.dirname(args.bag_file)
        # Get the base name of the input bag file (without extension)
        base_name = os.path.splitext(os.path.basename(args.bag_file))[0]
        # Construct the output JSON path in the same directory
        json_file_path = os.path.join(bag_dir, base_name + ".json")

    print(args.bag_file)
    bag_to_json(args.bag_file, json_file_path) 