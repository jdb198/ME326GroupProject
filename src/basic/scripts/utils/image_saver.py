import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')

        # Define QoS profile for direct subscriptions
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
                
        # Create a CvBridge to convert ROS image messages to OpenCV images
        self.bridge = CvBridge()
        
        # Create a subscriber for the image topic
        self.create_subscription(Image, '/locobot/camera/color/image_raw', self.image_callback, qos_profile)
        
        # Variable to count how many images we've saved
        self.image_count = 0
        
        # Create the directory for saving images
        if not os.path.exists('saved_images'):
            os.mkdir('saved_images')

    def image_callback(self, msg):
        if self.image_count >= 3:
            self.get_logger().info("Saved 10 images, stopping.")
            rclpy.shutdown()
            return

        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Save the image as a PNG file
            image_filename = f'saved_images/image_{self.image_count + 1}.png'
            cv2.imwrite(image_filename, cv_image)

            self.get_logger().info(f"Saved image {self.image_count + 1} as {image_filename}")
            self.image_count += 1

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
