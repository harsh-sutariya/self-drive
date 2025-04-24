#!/usr/bin/env python3

import rospy
import sys
from hybrid_navigation.srv import LoadTarget

def load_target_image(image_path):
    """Call the load target image service"""
    rospy.init_node('load_target_client')
    
    # Wait for the service to become available
    rospy.loginfo("Waiting for /vpr/load_target service...")
    try:
        rospy.wait_for_service('/vpr/load_target', timeout=10.0)
    except rospy.ROSException as e:
        rospy.logerr(f"Service /vpr/load_target did not appear within timeout: {e}")
        return
    
    try:
        # Create a service proxy
        load_target = rospy.ServiceProxy('/vpr/load_target', LoadTarget)
        
        # Call the service
        response = load_target(image_path)
        
        if response.success:
            rospy.loginfo(f"Successfully loaded target image: {image_path}")
        else:
            rospy.logerr(f"Failed to load target image: {image_path}")
            
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    # Get the image path from command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        load_target_image(image_path)
    else:
        rospy.logerr("No target image path provided.")
        sys.exit(1) 