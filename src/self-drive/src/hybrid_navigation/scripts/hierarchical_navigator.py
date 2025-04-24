#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Float32, Int32, String

class HierarchicalNavigator:
    def __init__(self):
        rospy.init_node('hierarchical_navigator')
        
        # Parameters
        self.linear_speed = rospy.get_param('~linear_speed', 0.2)  # m/s
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)  # rad/s
        self.distance_threshold = rospy.get_param('~distance_threshold', 0.3)  # m
        self.angle_threshold = rospy.get_param('~angle_threshold', 0.1)  # rad
        
        # PID controller parameters
        self.kp_linear = rospy.get_param('~kp_linear', 0.5)
        self.kp_angular = rospy.get_param('~kp_angular', 1.0)
        self.ki_linear = rospy.get_param('~ki_linear', 0.0)
        self.ki_angular = rospy.get_param('~ki_angular', 0.0)
        self.kd_linear = rospy.get_param('~kd_linear', 0.0)
        self.kd_angular = rospy.get_param('~kd_angular', 0.1)
        
        # State
        self.current_pose = None
        self.current_waypoint = None
        self.vpr_match = False
        self.vpr_similarity = 0.0
        self.target_reached = False
        
        # PID controller errors
        self.prev_linear_error = 0.0
        self.prev_angular_error = 0.0
        self.integral_linear_error = 0.0
        self.integral_angular_error = 0.0
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.target_reached_pub = rospy.Publisher('/navigation/target_reached', Bool, queue_size=10)
        self.status_pub = rospy.Publisher('/navigation/status', String, queue_size=10)
        
        # Subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.waypoint_sub = rospy.Subscriber('/topological_map/next_waypoint', PoseStamped, self.waypoint_callback)
        self.vpr_match_sub = rospy.Subscriber('/vpr/match', Bool, self.vpr_match_callback)
        self.vpr_similarity_sub = rospy.Subscriber('/vpr/similarity', Float32, self.vpr_similarity_callback)
        
        rospy.loginfo("Hierarchical Navigator initialized")
    
    def odom_callback(self, msg):
        """Store current odometry data"""
        self.current_pose = msg.pose.pose
        
        # If we have a waypoint, navigate to it
        if self.current_waypoint is not None and not self.target_reached:
            self.navigate_to_waypoint()
    
    def waypoint_callback(self, msg):
        """Handle new waypoint updates"""
        self.current_waypoint = msg.pose
        self.target_reached = False
        rospy.loginfo("New waypoint received")
    
    def vpr_match_callback(self, msg):
        """Handle visual place recognition match updates"""
        self.vpr_match = msg.data
        
        # If we have a match, we reached the target
        if self.vpr_match:
            self.target_reached = True
            self.stop_robot()
            self.target_reached_pub.publish(Bool(True))
            rospy.loginfo(f"Target reached based on visual recognition (similarity: {self.vpr_similarity})")
    
    def vpr_similarity_callback(self, msg):
        """Store latest visual place recognition similarity score"""
        self.vpr_similarity = msg.data
    
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
    
    def stop_robot(self):
        """Stop the robot by publishing zero velocity"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        
        # Publish several times to ensure the robot stops
        for _ in range(3):
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(0.1)
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Check if we've reached the target based on visual recognition
            if self.vpr_match:
                if not self.target_reached:
                    self.target_reached = True
                    self.stop_robot()
                    self.target_reached_pub.publish(Bool(True))
                    rospy.loginfo(f"Target reached based on visual recognition (similarity: {self.vpr_similarity})")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        navigator = HierarchicalNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        pass 