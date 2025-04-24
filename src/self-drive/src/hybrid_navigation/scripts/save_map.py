#!/usr/bin/env python3

import rospy
import os
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Int32

class MapSaver:
    def __init__(self):
        rospy.init_node('map_saver')
        
        # Create a service to save the map
        self.save_service = rospy.Service('/save_map', Trigger, self.save_map_callback)
        
        # Publisher to request map saving from topological mapper
        self.save_request_pub = rospy.Publisher('/topological_map/save_request', Int32, queue_size=1)
        
        rospy.loginfo("Map Saver initialized. Use 'rosservice call /save_map' to save the current map.")
    
    def save_map_callback(self, request):
        """Handle request to save the map"""
        try:
            # Publish a save request
            self.save_request_pub.publish(Int32(1))
            
            # Give time for the saving operation to complete
            rospy.sleep(2.0)
            
            return TriggerResponse(
                success=True,
                message="Map save request sent successfully. Check the logs for save status."
            )
        except Exception as e:
            rospy.logerr(f"Error while requesting map save: {e}")
            return TriggerResponse(
                success=False,
                message=f"Failed to request map save: {e}"
            )
    
    def run(self):
        """Main loop"""
        rospy.spin()

if __name__ == '__main__':
    try:
        saver = MapSaver()
        saver.run()
    except rospy.ROSInterruptException:
        pass 