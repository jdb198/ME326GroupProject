import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import rclpy.time
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from realsense2_camera_msgs.msg import Extrinsics
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker #Only for debugging

from cv_bridge import CvBridge
import tf2_ros

import numpy as np
import cv2 #Only for debugging
from scipy.spatial.transform import Rotation

from utils.align_depth_fncs import align_depth

from interbotix_xs_modules.xs_robot.locobot import InterbotixLocobotXS

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # TF Buffer and Listener
        buffer_length = rclpy.duration.Duration(seconds=10.0)
        self.tf_buffer = tf2_ros.Buffer(cache_time=buffer_length)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Define QoS profile for direct subscriptions
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # Subscribers
        # Locobot 1
        self.create_subscription(Image, '/locobot/camera/color/image_raw', self.image_callback, qos_profile)
        self.create_subscription(Image, '/locobot/camera/depth/image_rect_raw', self.depth_callback, qos_profile)
        self.create_subscription(CameraInfo, '/locobot/camera/color/camera_info', self.camera_info_callback, 10)
        self.create_subscription(CameraInfo, '/locobot/camera/depth/camera_info', self.camera_depth_info_callback, 10)
        # Locobot 3
        # self.create_subscription(Image, '/locobot/camera/color/image_raw', self.image_callback, qos_profile)
        # self.create_subscription(Image, '/locobot/camera/depth/image_rect_raw', self.depth_callback, qos_profile)
        # self.create_subscription(CameraInfo, '/locobot/camera/color/camera_info', self.camera_info_callback, 10)
        # self.create_subscription(CameraInfo, '/locobot/camera/depth/camera_info', self.camera_depth_info_callback, 10)
        # Simulation
        # self.create_subscription(Odometry, "/locobot/odom", self.odom_callback, 10)
        self.create_subscription(Odometry, "/locobot/mobile_base/odom", self.odom_callback, qos_profile)
        
        self.prompt_sub = self.create_subscription(String, "/perception/object", self.prompt_callback, 10)

        self.timer = self.create_timer(0.1, self.process_if_ready)

        self.bridge = CvBridge()

        # self.object_detector = ...

        # Frames
        # Locobot 1
        self.base_link_frame = "locobot/base_link"
        self.camera_frame = "camera_color_frame"
        self.depth_camera_frame = "camera_depth_frame"
        # Locobot 3
        # self.camera_frame = "camera_color_optical_frame" #"camera_locobot_link"
        # self.depth_camera_frame = "camera_depth_optical_frame"

        # State variables
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_depth_camera_info = None
        self.latest_odom = None
        self.current_prompt = None
        self.latest_depth2cam_extrinsic = None

        # Debugging Publishers
        self.debug_image_pub = self.create_publisher(Image, "/perception/debug_image", 10)
        self.marker_pub = self.create_publisher(Marker, "/perception/debug_marker", 10)

        self.locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s")
        self.locobot.arm.go_to_home_pose()
        self.get_logger().info("Perception Node Successfully created")

    def image_callback(self, msg):
        self.latest_rgb = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def camera_depth_info_callback(self, msg):
        self.latest_depth_camera_info = msg

    def depth2cam_ext_callback(self, msg):
        self.latest_depth2cam_extrinsic = msgdef odom_callback(self, msg):
    def odom_callback(self, msg):
        self.latest_odom = msg
    
    def prompt_callback(self, msg):
        """ Store the user-given prompt and trigger processing """
        self.get_logger().info(f"Received prompt: {msg.data}")
        self.current_prompt = msg.data

    def process_if_ready(self):
        """ Check if all data is synchronized and process if a prompt is given """
        # if not self.current_prompt:
        #     return  # Skip processing if no prompt is set

        if not all([self.latest_rgb, self.latest_depth, self.latest_camera_info, self.latest_odom]):
        # for Locobot 3
        # if not all([self.latest_rgb, self.latest_depth, self.latest_camera_info]):
            self.get_logger().warn("Waiting for required messages to be prepared...")
            if not self.latest_rgb:
                self.get_logger().warn("Waiting for latest_rgb")
            if not self.latest_depth:
                self.get_logger().warn("Waiting for latest_depth")
            if not self.latest_camera_info:
                self.get_logger().warn("Waiting for latest_camera_info")
            # if not self.latest_depth2cam_extrinsic:
            #     self.get_logger().warn("Waiting for latest_depth2cam_extrinsic")
            return  # Wait until all necessary data is available

        # Pass copies of the latest data to avoid changes during processing
        rgb_msg = self.latest_rgb
        depth_msg = self.latest_depth
        camera_info_msg = self.latest_camera_info
        odom_msg = self.latest_odom
        prompt = self.current_prompt
        
        # Process image with these copies
        self.process_image(rgb_msg, depth_msg, camera_info_msg, odom_msg, prompt)

    def process_image(self, rgb_msg, depth_msg, camera_info_msg, odom_msg, prompt):
        self.get_logger().info("Image Processing Started")
        timestamp = rgb_msg.header.stamp
        # Get transform from camera frame to base link frame
        camera_to_base = self.get_transform(self.camera_frame, self.base_link_frame, timestamp)
        if not camera_to_base:
            print("Failed to find the desired camera_to_base tf. Will try later....")
            return
        
        print("RGB image reference frame: ", rgb_msg.header.frame_id)
        print("Depth image reference frame: ", depth_msg.header.frame_id)
        depth_to_rgb = self.get_transform(self.depth_camera_frame, self.camera_frame, timestamp)
        if not depth_to_rgb:
            print("Failed to find the desired depth_to_rgb tf. Will try later....")
            return

        # Convert images
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        
        # Run VLM (??)
        # TODO Get pixel values from VLM

        # For Debugging
        pixel_x, pixel_y = [400, 240]
        debug_image = rgb_image.copy()
        cv2.circle(debug_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
        debug_img_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
        debug_img_msg.header = rgb_msg.header
        self.debug_image_pub.publish(debug_img_msg)

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")   # Depth in milimeters
        # Depth-RGB alignment
        # depth_image = self.get_depth_aligned_with_rgb(depth_image, rgb_image, self.transform_to_matrix(self.latest_depth2cam_extrinsic))
        depth_image = self.get_depth_aligned_with_rgb(depth_image, rgb_image, self.transform_to_matrix(depth_to_rgb.transform))
        
        depth = depth_image[int(pixel_y), int(pixel_x)]/1000.0
        camera_coords = self.pixel_to_camera(pixel_x, pixel_y, depth, camera_info_msg)
        self.get_logger().info(f"Converted Camera Coordinates: {camera_coords[0]}, {camera_coords[1]}, {camera_coords[2]}")
        # camera_coords = np.array([camera_coords[2], -1 * camera_coords[0], -1 * camera_coords[1]]) # to manually convert in simulation

        timestamp = rgb_msg.header.stamp
        base_pose = odom_msg.pose.pose
        print("Robot base pose: " base_pose)
        world_coords = self.camera_to_world(camera_coords, base_pose, camera_to_base)

        self.get_logger().info(f"Converted World Coordinates: {world_coords[0]}, {world_coords[1]}, {world_coords[2]}")
        self.publish_debug_marker(world_coords)

        # Clear the prompt after processing
        self.current_prompt = None

    def get_transform(self, ref_frame, target_frame, timestamp = self.latest_rgb.header.stamp, timeout_margin = 1.0):
        try:
            # Get transform from ref_frame to target_frame
            print("Finding transformation at :", timestamp, " from ", ref_frame, " to ", target_frame)
            print("Checking transform availability: ", self.tf_buffer.can_transform(target_frame, ref_frame, timestamp))
            ref_to_target = self.tf_buffer.lookup_transform(target_frame, ref_frame, timestamp, rclpy.duration.Duration(seconds=timeout_margin))
            print("Successfully found Transform")
            return ref_to_target
        except tf2_ros.LookupException:
            self.get_logger().warn("Transform not found.")
            return None
        except tf2_ros.ConnectivityException:
            self.get_logger().warn("Connectivity issue.")
            return None
        except tf2_ros.ExtrapolationException:
            self.get_logger().warn("Extrapolation issue.")
            return None
    
    def get_depth_aligned_with_rgb(self, depth_img, rgb_img, cam2cam_transform = np.eye(4)):
        depth_K = (self.latest_depth_camera_info.k[0], self.latest_depth_camera_info.k[4], self.latest_depth_camera_info.k[2], self.latest_depth_camera_info.k[5])
        rgb_K = (self.latest_camera_info.k[0], self.latest_camera_info.k[4], self.latest_camera_info.k[2], self.latest_camera_info.k[5])
        return align_depth(depth_img, depth_K, rgb_img, rgb_K, cam2cam_transform)

    def pixel_to_camera(self, pixel_x, pixel_y, depth, camera_info_msg):
        """ Convert pixel coordinates to camera frame (3D point) """
        # fx = 620 # Focal length x
        # fy = 620 # Focal length y
        # cx = 320 # Principal point x
        # cy = 240 # Principal point y

        fx = camera_info_msg.k[0]  # Focal length x
        fy = camera_info_msg.k[4]  # Focal length y
        cx = camera_info_msg.k[2]  # Principal point x
        cy = camera_info_msg.k[5]  # Principal point y

        print(fx, fy, cx, cy)

        # Compute 3D position in camera frame
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth
        return np.array([x, y, z])

    def camera_to_world(self, camera_coords, base_pose, camera_to_base):
        """ Convert camera coordinates to world coordinates using /locobot/odom and TF """
        # Convert odometry pose (base_link in locobot/odom) to matrix
        base_to_odom_matrix = self.pose_to_matrix(base_pose)

        # Convert base_link → camera_link transform to matrix
        camera_to_base_matrix = self.transform_to_matrix(camera_to_base)

        # Compute camera-to-world transformation
        camera_to_world_matrix = base_to_odom_matrix @ camera_to_base_matrix

        # Convert camera point to world frame
        camera_coords_homogeneous = np.append(camera_coords, 1)  # Convert to (x, y, z, 1)
        world_coords = camera_to_world_matrix @ camera_coords_homogeneous
        return world_coords[:3]  # Extract (x, y, z)

    def pose_to_matrix(self, pose):
        """ Convert a geometry_msgs/Pose to a 4x4 transformation matrix """
        translation = np.array([pose.position.x, pose.position.y, pose.position.z])
        quaternion = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

        # Create 4x4 transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix

    def transform_to_matrix(self, transform):
        """ Convert a geometry_msgs/TransformStamped to a 4x4 transformation matrix """
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])
        quaternion = np.array([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ])
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

        # Create 4x4 transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix

    def publish_debug_marker(self, world_coords):
        """ Publish a marker in RViz at the transformed world coordinates """
        marker = Marker()
        marker.header.frame_id = self.base_link_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "debug"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set marker position
        marker.pose.position.x = world_coords[0]
        marker.pose.position.y = world_coords[1]
        marker.pose.position.z = world_coords[2]

        marker.pose.orientation.w = 1.0  # No rotation

        # Set marker scale (size)
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        # Set color (red)
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
