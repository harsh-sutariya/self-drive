#!/usr/bin/env python3

import rospy
import os
import rospkg

def print_bag_path():
    # Initialize the node
    rospy.init_node('print_bag_path', anonymous=True)
    
    # Get the bag filename from parameter server
    bag_filename = rospy.get_param('~bag_filename', 'default.bag')
    
    # Get the package path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('exploration')
    
    # Calculate the workspace root (4 levels up from the package)
    workspace_root = os.path.abspath(os.path.join(pkg_path, '..', '..', '..', '..'))
    
    # Construct the full path
    full_path = os.path.join(workspace_root, bag_filename)
    
    # Print the path
    rospy.loginfo("Bag will be saved at: %s", full_path)

if __name__ == '__main__':
    try:
        print_bag_path()
    except rospy.ROSInterruptException:
        pass 