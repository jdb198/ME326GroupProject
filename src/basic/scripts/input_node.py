#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.img = Image()
        # Define QoS profile with best effort reliability
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
        # Create subscription with best effort QoS profile
        self.cam_sub = self.create_subscription(
            Image,
            "/locobot/camera_frame_sensor/image_raw",
            self.cam_callback,
            qos_profile
        )
        self.cam_pub = self.create_publisher(Image, "/locobot/camera_frame_sensor/image_subscribed", 10)
        self.publisher_ = self.create_publisher(Twist, '/locobot/diffdrive_controller/cmd_vel_unstamped', 10)
        self.timer = self.create_timer(0.1, self.publish_velocity)  # Publish every 0.1 seconds
        self.get_logger().info('CmdVelPublisher node has started')

    def cam_callback(self, msg: Image) -> None:
        print("here")
        self.get_logger().info(f'Subscribed Image: {msg.height}')
        self.img = msg
        self.get_logger().info('Subscribed Image Input')
    
    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = 0.5  # Set linear velocity (forward)
        msg.angular.z = 0.5  # Set angular velocity (turn)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: linear.x={msg.linear.x}, angular.z={msg.angular.z}')
        self.cam_pub.publish(self.img)

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
