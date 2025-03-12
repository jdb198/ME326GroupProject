#!/usr/bin/env python3
from interbotix_xs_modules.xs_robot.locobot import InterbotixLocobotXS
import numpy as np
import time

import rclpy
import rclpy.duration
from rclpy.node import Node
from basic.msg import TargetObject
from std_msgs.msg import Bool

# THIS SCRIPT DOESN'T WORK WITHOUT FIXING THE PUBLISHERS AND SUBSCRIBERS. 
# ALSO PUT IN THE CALL FOR THE FINGER DISTANCE 

class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node')

        self.locobot_type = 1 # 0(sim), 1, 3
        self.offset_dict = {1: [-0.2, 0.0, 0.06], 3: [0.0, 0.0, 0.04]}
        self.offset = self.offset_dict[self.locobot_type]
        self.debug = True

        # Subscribers
        self.create_subscription(TargetObject, "/perception/target_coord", self.coord_callback, 10)

        # State variables

        # Publisher
        self.grasp_success_pub = self.create_publisher(Bool, "/manipulation/grasp_success", 10)

        # Locobot instance for debugging purpose
        self.locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s")
        self.locobot.gripper.release()
        self.locobot.arm.go_to_sleep_pose()
        time.sleep(1.0)

        self.get_logger().info("Manipulation Node Successfully created")
    
    def publish_grasp_success(self, success):
        pub_msg = Bool()
        pub_msg.data = success
        self.grasp_success_pub.publish(pub_msg)
    
    def coord_callback(self, msg):
        """ Store the image_segment_node-given object information and trigger processing """
        if msg.purpose is not 1:
            return
        self.get_logger().info(f"Received new target object information, {msg.x, msg.y, msg.z}")
        self.locobot.arm.go_to_home_pose()
        time.sleep(0.5)
        self.locobot.gripper.release()
        self.locobot.arm.set_ee_pose_components(x=msg.x + self.offset[0], y=msg.y + self.offset[0], z= max(msg.z + self.offset[0], 0.02), roll=0.0, pitch=1.5)
        time.sleep(0.3)
        self.locobot.gripper.grasp()
        time.sleep(3.0)
        finger_position = self.locobot.gripper.get_finger_position()

        if finger_position > .02: # this is telling us if it is grasping or not. 
            print('Success')
            self.locobot.arm.go_to_home_pose()
            self.publish_grasp_success(True)
            return

        print('Failure')
        self.publish_grasp_success(False)
        input('Failed to grasp... measure your offset')
        self.locobot.arm.go_to_sleep_pose()

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()