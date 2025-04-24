#!/usr/bin/env python3

import rospy
import cv2
import os
from std_msgs.msg import String
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BestMatchViewer:
    def __init__(self):
        rospy.init_node('best_match_viewer', anonymous=True)
        
        # State
        self.best_match_path = None
        self.best_match_image = None
        self.target_image = None
        self.target_image_path = rospy.get_param('~target_image_path', None)
        
        # Setup figure for real-time visualization
        plt.ion()  # Enable interactive mode
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('Visual Place Recognition Matching')
        self.axes[0].set_title('Target Image')
        self.axes[1].set_title('Best Match')
        
        # Load target image if available
        if self.target_image_path and os.path.exists(self.target_image_path):
            self.target_image = cv2.imread(self.target_image_path)
            if self.target_image is not None:
                self.target_image = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2RGB)
                self.axes[0].imshow(self.target_image)
                self.axes[0].axis('off')
                plt.pause(0.01)
        
        # Subscribe to best match path updates
        self.path_sub = rospy.Subscriber('/vpr/best_match_path', String, self.path_callback)
        
        rospy.loginfo("Best Match Viewer initialized")
        
    def path_callback(self, msg):
        """Handle new best match path updates"""
        self.best_match_path = msg.data
        rospy.loginfo(f"New best match path received: {self.best_match_path}")
        
        # Load and display the new best match
        if os.path.exists(self.best_match_path):
            self.best_match_image = cv2.imread(self.best_match_path)
            if self.best_match_image is not None:
                self.best_match_image = cv2.cvtColor(self.best_match_image, cv2.COLOR_BGR2RGB)
                self.update_display()
    
    def update_display(self):
        """Update the matplotlib display with new images"""
        if self.best_match_image is not None:
            self.axes[1].clear()
            self.axes[1].imshow(self.best_match_image)
            self.axes[1].set_title(f'Best Match\nPath: {os.path.basename(self.best_match_path)}')
            self.axes[1].axis('off')
            
            # Force redraw
            self.fig.canvas.draw_idle()
            plt.pause(0.01)
    
    def run(self):
        """Main loop"""
        plt.show(block=True)  # Show plot and block

if __name__ == '__main__':
    try:
        viewer = BestMatchViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass 