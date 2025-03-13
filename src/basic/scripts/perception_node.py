#!/usr/bin/env python3

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import rclpy.time
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import Image, CameraInfo,PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from realsense2_camera_msgs.msg import Extrinsics
from geometry_msgs.msg import TransformStamped, PoseStamped
from std_msgs.msg import ByteMultiArray, UInt8MultiArray, String, Float32MultiArray
from visualization_msgs.msg import Marker #Only for debugging
from basic.msg import TargetObject

from cv_bridge import CvBridge
import tf2_ros

import numpy as np
import cv2 #Only for debugging
from scipy.spatial.transform import Rotation

from utils.align_depth_fncs import align_depth

from interbotix_xs_modules.xs_robot.locobot import InterbotixLocobotXS
import time
import struct

import torch
import numpy as np 
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from std_msgs.msg import ByteMultiArray, UInt8MultiArray, String, Float32MultiArray
from PIL import Image as pil_image
import tf2_ros
import torch
from std_msgs.msg import Bool
import cv2
from ultralytics import YOLO
from open_clip import create_model, tokenize  # Assuming these are defined elsewhere
import os

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.locobot_type = 1 # 0(sim), 1, 3
        self.debug = True

        rgb_info_topics = {0: '/locobot/camera/camera_info', 1: '/locobot/camera/color/camera_info', 3: '/locobot/camera/camera/color/camera_info'}
        depth_info_topics = {0: '/locobot/camera/depth/camera_info', 1: '/locobot/camera/color/camera_info', 3: '/locobot/camera/camera/depth/camera_info'}
        rgb_img_topics = {0: '/locobot/camera/image_raw', 1: '/locobot/camera/color/image_raw', 3: '/locobot/camera/camera/color/image_raw'}
        depth_img_topics = {0: '/locobot/camera/depth/image_raw', 1: '/locobot/camera/depth/image_rect_raw', 3: '/locobot/camera/camera/depth/image_rect_raw'}
        odom_topics = {0:'/locobot/sim_ground_truth_pose', 1: "/locobot/mobile_base/odom", 3: "/locobot/mobile_base/odom"}

        # TF Buffer and Listener
        buffer_length = rclpy.duration.Duration(seconds=10.0)
        self.tf_buffer = tf2_ros.Buffer(cache_time=buffer_length)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Define QoS profile for direct subscriptions
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # Subscribers
        print(rgb_img_topics[self.locobot_type])
        self.create_subscription(Image, rgb_img_topics[self.locobot_type], self.image_callback, qos_profile)
        self.create_subscription(Image, depth_img_topics[self.locobot_type], self.depth_callback, qos_profile)
        self.create_subscription(Odometry, odom_topics[self.locobot_type], self.odom_callback, qos_profile)
        # self.create_subscription(TargetObject, "object", self.object_callback, 10)
        self.create_subscription(Bool, "/manipulation/grasp_success", self.grasp_callback, 10)
        

        # Subscribe to transcribed text
        self.text_subscription = self.create_subscription(String, 'transcribed_text', self.text_callback, 10)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info("Image segment node started. Waiting for transcribed text...")

        self.timer = self.create_timer(0.1, self.process_if_ready)
        self.bridge = CvBridge()

        # Frames
        self.base_link_frame = "locobot/base_link"
        self.arm_base_link_frame = "locobot/arm_base_link"
        self.camera_frame = "camera_color_optical_frame" if self.locobot_type > 0 else "locobot/camera_depth_link"
        self.depth_camera_frame = "camera_depth_optical_frame" if self.locobot_type > 0 else "locobot/camera_depth_link"

        # State variables
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_odom = None
        self.target_pixel = None
        self.need_world_coordinate = True # Always False for Locobot 3
        self.image_path = None
        self.saved_world_coords = None
        self.text_msg = None
        self.fail_count = 0
        self.YOLO_called = False
        
        self.get_logger().info("Waiting for RGB Camera Info....")
        self.rgb_camera_info = wait_for_message(CameraInfo, self, rgb_info_topics[self.locobot_type])[1]
        self.get_logger().info("Success!")
        print(self.rgb_camera_info)

        self.get_logger().info("Waiting for Depth Camera Info....")
        self.depth_camera_info = wait_for_message(CameraInfo, self, depth_info_topics[self.locobot_type])[1]
        self.get_logger().info("Success!")
        print(self.depth_camera_info)

        # self.get_logger().info("Waiting for Depth to RGB TF...")
        self.depth_to_rgb_tf = None
        # while not self.depth_to_rgb_tf:
        #     self.depth_to_rgb_tf = self.get_transform(self.depth_camera_frame, self.camera_frame)
        # self.get_logger().info("Success!")
        # print(self.depth_to_rgb_tf)
        # self.get_logger().info("Waiting for Depth to RGB Extrinsic...")
        # self.depth2cam_extrinsic = wait_for_message('/locobot/camera/extrinsics/depth_to_color', Extrinsics, self)
        # self.get_logger().info("Success!")

        # Publisher
        #self.target_coord_pub = self.create_publisher(PoseStamped, "/perception/target_coord", 10)
        self.target_coord_pub = self.create_publisher(TargetObject, "/perception/target_coord", 10)

        # Debugging Publishers
        self.debug_image_pub = self.create_publisher(Image, "/perception/debug_image", 10)
        self.debug_depth_pub = self.create_publisher(Image, "/perception/debug_depth", 10)
        self.marker_pub = self.create_publisher(Marker, "/perception/debug_marker", 10)
        self.pc_pub = self.create_publisher(PointCloud2, "/perception/debug_pc", 10)

        # Locobot instance for debugging purpose
        # if self.debug and self.locobot_type > 0:
        #     self.locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s")
        #     # Comment this if you do not need to tune the offset values anymore
        #     self.locobot.gripper.release()
        #     self.locobot.arm.go_to_sleep_pose()
        #     time.sleep(1.0)

        self.get_logger().info("Perception Node Successfully created")

    # def next_step_callback(self, msg):
    #     self.needs_new_image = msg 

    def image_callback(self, msg):
        self.latest_rgb = msg
        # Save image frame for image segmentation
        self.save_camera_frame(msg)

    def depth_callback(self, msg):
        self.latest_depth = msg

    def odom_callback(self, msg):
        self.latest_odom = msg

    def grasp_callback(self, msg):
        if msg.data:
            target_msg = TargetObject()
            target_msg.x = 0.0
            target_msg.y = 0.0
            target_msg.z = 0.0

            target_msg.pose = PoseStamped()
            target_msg.pose.pose.position.x = 0.0
            target_msg.pose.pose.position.y = 0.0
            target_msg.pose.pose.position.z = 0.0
            target_msg.pose.pose.orientation.x = 0.0
            target_msg.pose.pose.orientation.y = 0.0
            target_msg.pose.pose.orientation.z = 0.0
            target_msg.pose.pose.orientation.w = 1.0 # cos(theta/2)
            target_msg.purpose = 0
            self.target_coord_pub.publish(target_msg)
    
    # def object_callback(self, msg):
    #     """ Store the image_segment_node-given object information and trigger processing """
    #     self.get_logger().info(f"Received new target object information")
    #     #self.target_pixel = [msg.x, msg.y]
    #     self.need_world_coordinate = True if msg.purpose == 0 else False

    def text_callback(self, msg):
        """Process the received transcribed text and use it as a prompt for object detection"""
        # if self.YOLO_called:
        #     self.get_logger().warn(f"Received text prompt: {msg.data}")
        #     self.target_pixel = (0, 0)
        #     return

        self.text_msg = msg
        text_prompt = msg.data
        self.get_logger().info(f"Received text prompt: {text_prompt}")
        # if text_prompt[-1] == '.':
        #     text_prompt = text_prompt[:-1]
        text_prompt = text_prompt.lower()
        print("Parsed Text: ",text_prompt)
        
        try:
            # Ensure we have a valid image
            if not self.image_path:
                raise ValueError("No camera frame available for processing.")
            
            # Get object center coordinates based on text prompt
            center_x, center_y = self.get_center(self.image_path, text_prompt)
            print(center_x)
            print(center_y)
            
            # Store target pixel for later processing in process_image()
            self.target_pixel = (int(center_x), int(center_y))
            print("VLM gave", self.target_pixel)
            
            self.get_logger().info(f"Updated target pixel: ({center_x}, {center_y})")
            self.YOLO_called = True
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def save_camera_frame(self, msg):
        try:
            # If an image has already been saved, skip saving again
            if self.image_path:
                return
            
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Save the image to disk
            self.image_path = "/tmp/perception_frame.jpg"  # Change this path if needed
            cv2.imwrite(self.image_path, cv_image)

            self.get_logger().info(f"Saved camera frame to {self.image_path}")
        
        except Exception as e:
            self.get_logger().error(f"Failed to save camera frame: {str(e)}")


    def preprocess_pil_image(self, image):
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return image_tensor

    def get_center(self, image_path, text_prompt):
        """Find the center of the object in the image that best matches the text prompt"""
        # Load the image
        image = pil_image.open(image_path).convert("RGB")

        # Load YOLOv11 model (pre-trained on COCO dataset)
        model = YOLO("yolo11x.pt")  # YOLOv11 model
        results = model(image_path)  # Perform inference

        # Extract detection results from the tensor
        detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data as NumPy array
        print(detections)
        boxes = detections[:, :4]  # Bounding box coordinates
        scores = detections[:, 4]  # Confidence scores
        classes = detections[:, 5].astype(int)  # Class indices

        # Access class names from the YOLOv11 model
        class_names = model.names  # Mapping class indices to names

        # Load CLIP model
        clip_model = create_model("ViT-B-32", pretrained="openai")
        clip_model.eval()

        clip_model = clip_model.to(self.device)

        # Filter detections with CLIP and print debug information
        filtered_boxes = []
        confidences = []
        similarities = []

        image = results[0].orig_img
        # convert to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Image for Classification", image)

        for i, box in enumerate(boxes):
            # Crop object from the image
            xmin, ymin, xmax, ymax = map(int, box)  # Ensure indices are integers
            # cropped_object = image.crop((xmin, ymin, xmax, ymax))
            cropped_object = image[ymin:ymax, xmin:xmax]
            # convert back to pil
            cropped_object = pil_image.fromarray(cropped_object)

            # Preprocess the cropped object and text
            image_input = self.preprocess_pil_image(cropped_object)
            text_input = tokenize([text_prompt]).to(self.device)

            # Compute similarity using CLIP
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_input)
                similarity = torch.cosine_similarity(image_features, text_features).item()
            confidences.append(scores[i])
            similarities.append(similarity)

        # Determine max clip similarity (similarity with name), this will grab multiple. it's supposed to
        max_similarity = max(similarities)
        similarity_indices = []
        for i, similarity in enumerate(similarities):
            if abs(similarity - max_similarity) < .005:  # set a tolerance for picking the best thing
                similarity_indices.append(i)

        ### Pull confidences for max similarities
        confidences_max_similarities = [confidences[i] for i in similarity_indices]

        max_confidence = max(confidences_max_similarities)  ### Using best YOLO confidence
        max_confidence_index = confidences.index(max_confidence)
        filtered_boxes.append((boxes[max_confidence_index], class_names[classes[max_confidence_index]],
                               scores[max_confidence_index], similarities[max_confidence_index]))

        ## Find the Center of the best box
        center_x = (boxes[max_confidence_index][0] + boxes[max_confidence_index][2]) / 2
        center_y = (boxes[max_confidence_index][1] + boxes[max_confidence_index][3]) / 2

        return center_x, center_y

    def process_if_ready(self):
        """ Check if all data is synchronized and process if a target pixel is given """
        if not self.target_pixel:
            return  # Skip processing if there is no target pixel to transform
        if not all([self.latest_rgb, self.latest_depth, self.rgb_camera_info, not self.need_world_coordinate or self.latest_odom]):
            self.get_logger().warn("Waiting for required messages to be prepared...")
            if not self.latest_rgb:
                self.get_logger().warn("Waiting for latest_rgb")
            if not self.latest_depth:
                self.get_logger().warn("Waiting for latest_depth")
            if not self.rgb_camera_info:
                self.get_logger().warn("Waiting for rgb_camera_info")
            if self.need_world_coordinate and not self.latest_odom:
                self.get_logger().warn("Waiting for latest_odom")
            # if not self.depth2cam_extrinsic:
            #     self.get_logger().warn("Waiting for depth2cam_extrinsic")
            return  # Wait until all necessary data is available
        
        # Pass copies of the latest data to avoid changes during processing
        rgb_msg = self.latest_rgb
        depth_msg = self.latest_depth
        camera_info_msg = self.rgb_camera_info
        odom_msg = self.latest_odom
        
        # Process image with these copies
        self.process_image(rgb_msg, depth_msg, camera_info_msg, odom_msg)
        # self.needs_new_image = False 

    def process_image(self, rgb_msg, depth_msg, camera_info_msg, odom_msg):
        self.get_logger().info("Image Processing Started")
        # Get transform from camera frame to desired link frame
        # timestamp = rgb_msg.header.stamp #getting tf at the exact timestamp is hard in the real robot
        camera_to_base = self.get_transform(self.camera_frame, self.base_link_frame) # pose of camera relative to base
        if self.need_world_coordinate and not camera_to_base:
            print("Failed to find the desired camera_to_base tf. Will try later....")
            return

        camera_to_arm_base = self.get_transform(self.camera_frame, self.arm_base_link_frame) # pose of camera relative to arm base
        base_to_arm_base = self.get_transform(self.base_link_frame, self.arm_base_link_frame) # pose of base relative to arm_base
        if not self.need_world_coordinate and (not camera_to_arm_base or not base_to_arm_base):
            print("Failed to find the desired camera_to_arm__base or base_to_arm_base tf. Will try later....")
            return

        # print("RGB image reference frame: ", rgb_msg.header.frame_id)
        # print("Depth image reference frame: ", depth_msg.header.frame_id)
        # depth_to_rgb = self.get_transform(self.depth_camera_frame, self.camera_frame, timestamp) # finding tf at exact timestamp is hard
        # depth_to_rgb = self.get_transform(self.depth_camera_frame, self.camera_frame)
        self.depth_to_rgb_tf = self.get_transform(self.depth_camera_frame, self.camera_frame) # pose of depth relative to color frame
        # if self.locobot_type > 0 and not depth_to_rgb:
        #     print("Failed to find the desired depth_to_rgb tf. Will try later....")
        #     return

        pixel_x, pixel_y = self.target_pixel
        
        # Convert images
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        # Depth-RGB alignment
        if self.locobot_type > 0:
            # depth_image = self.get_depth_aligned_with_rgb(depth_image, rgb_image, np.linalg.inv(self.transform_to_matrix(self.depth_to_rgb_tf)))
            depth_image = self.get_depth_aligned_with_rgb(depth_image, rgb_image, self.transform_to_matrix(self.depth_to_rgb_tf))
        
        if self.debug:
            debug_image = rgb_image.copy()
            cv2.circle(debug_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
            image_filename = f'saved_images/image_{pixel_x}_{pixel_y}.png'
            cv2.imwrite(image_filename, debug_image)
            debug_img_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_img_msg.header = rgb_msg.header
            self.debug_image_pub.publish(debug_img_msg)

            debug_depth_img = depth_image.copy()
            cv2.circle(debug_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
            debug_depth_msg = self.bridge.cv2_to_imgmsg(debug_depth_img, "16UC1")
            debug_depth_msg.header = depth_msg.header
            self.debug_depth_pub.publish(debug_depth_msg)
        
        depth = depth_image[int(pixel_y), int(pixel_x)]/1000.0 if self.locobot_type > 0 else depth_image[int(pixel_y), int(pixel_x)]
        camera_coords = self.pixel_to_camera(pixel_x, pixel_y, depth, camera_info_msg)
        self.get_logger().info(f"Converted Camera Coordinates: {camera_coords[0]}, {camera_coords[1]}, {camera_coords[2]}")
        # if self.locobot_type == 0:
        #     camera_coords = np.array([camera_coords[2], -1 * camera_coords[0], -1 * camera_coords[1]]) # to manually convert to match frame configuration in simulation

        #self.target_pixel = None  # Clear target pixel after processing
        
        if self.need_world_coordinate:
            # camera_coords_homogeneous = np.append(camera_coords, 1)  # Convert to (x, y, z, 1)
            # camera_to_base_matrix = self.transform_to_matrix(camera_to_base) #base_T_camera
            # base_coords = camera_to_base_matrix @ camera_coords_homogeneous
            # self.get_logger().info(f"Converted Base Coordinates: {base_coords[0]}, {base_coords[1]}, {base_coords[2]}")
            # if base_coords[0] < 0.3 or base_coords[0] > 0.8 or base_coords[1] > 0.2 or base_coords[1] < -0.1 or base_coords[2] > 0.1 or base_coords[2] < 0.0:
            #     print("Unexpected Output... Will run YOLO again")
            #     self.text_callback(self.text_msg)
            #     self.fail_count += 1
            #     if self.fail_count < 1:
            #         return
            #     base_coords[1] = 0.0
            # self.fail_count=0
            # self.publish_target_coord(base_coords)
            # self.need_world_coordinate = False
            base_pose = odom_msg.pose.pose
            print("Robot base pose: ", base_pose)
            world_coords = self.camera_to_world(camera_coords, base_pose, camera_to_base)
            self.get_logger().info(f"Converted World Coordinates: {world_coords[0]}, {world_coords[1]}, {world_coords[2]}")
            if world_coords[0] < 0.3 or world_coords[0] > 0.8 or world_coords[1] > 0.2 or world_coords[1] < -0.1 or world_coords[2] > 0.1 or world_coords[2] < 0.0:
                print("Unexpected Output... Will try transformation again")
                # self.text_callback(self.text_msg)
                self.fail_count += 1
                if self.fail_count < 10:
                    return
                world_coords[1] = 0.0 # kinda hard-coded stuff
            # world_coords[1] = 0.0 
            # self.saved_world_coords = world_coords
            self.fail_count=0
            self.publish_target_coord(world_coords)
            self.need_world_coordinate = False
        else:
            # second call: ignore the second YOLO Call and just use the previous value
            if self.saved_world_coords is not None:
                time.sleep(2.0)
                world_pose = odom_msg.pose.pose
                world_T_base = self.pose_to_matrix(world_pose)
                base_T_world = np.linalg.inv(world_T_base)
                saved_coords_hom = np.append(self.saved_world_coords, 1)
                print(saved_coords_hom)
                print("Will use saved coordinate, ", self.saved_world_coords)
                armbase_T_base = self.transform_to_matrix(base_to_arm_base)
                armbase_T_target = armbase_T_base @ base_T_world @ saved_coords_hom
                arm_base_coords = armbase_T_target[:3]
            else:
                arm_base_coords = self.camera_to_arm_base(camera_coords, camera_to_arm_base)

            self.get_logger().info(f"Converted Arm Base Coordinates: {arm_base_coords[0]}, {arm_base_coords[1]}, {arm_base_coords[2]}")
            
            # Temporary fix to avoid using critically inaccurate depth values
            if arm_base_coords[0] < 0.2 or arm_base_coords[0] > 0.5 or arm_base_coords[1] < -0.1 or arm_base_coords[1] > 0.1 or arm_base_coords[2] < -0.2 or arm_base_coords[0] < 0.2:
                print("Transformed the coordinate, but unexpected value. Will try transformation again....")
                # self.text_callback(self.text_msg)
                self.fail_count += 1
                if self.fail_count < 5:
                    return
            self.fail_count=0
            
            self.publish_target_coord(arm_base_coords)
            # Comment this if you do not need to tune the offset values anymore
            if self.debug:
                # Change these offset values! (can be different depending on the locobot)
                x_offset = 0.0
                y_offset = 0.0
                z_offset = 0.0
                # input("Waiting for robot arm movement confirmation")
                # self.locobot.gripper.release()
                # self.locobot.arm.set_ee_pose_components(x=base_coords[0] + x_offset, y=base_coords[1]+y_offset, z=base_coords[2]+z_offset, roll=0.0, pitch=1.0)
                # self.locobot.gripper.grasp()

        # For debug purpose
        if self.debug:
            camera_coords_homogeneous = np.append(camera_coords, 1)  # Convert to (x, y, z, 1)
            camera_to_base_matrix = self.transform_to_matrix(camera_to_base)
            base_coords = camera_to_base_matrix @ camera_coords_homogeneous
            self.get_logger().info(f"Converted Base Coordinates: {base_coords[0]}, {base_coords[1]}, {base_coords[2]}")
            self.publish_debug_marker(base_coords[:3])
            # self.publish_pointcloud(rgb_image, depth_image)

        # Empty the current prompt to avoid running image processing again
        self.target_pixel = None
        self.image_path = None

    def get_transform(self, ref_frame, target_frame, timestamp = rclpy.time.Time(), timeout_margin = 1.0):
        try:
            # Get transform from ref_frame to target_frame
            ref_to_target = self.tf_buffer.lookup_transform(target_frame, ref_frame, timestamp)
            return ref_to_target
        except tf2_ros.LookupException:
            print("Tried finding transformation at :", timestamp, " from ", ref_frame, " to ", target_frame)
            self.get_logger().warn("Transform not found.")
            return None
        except tf2_ros.ConnectivityException:
            print("Tried finding transformation at :", timestamp, " from ", ref_frame, " to ", target_frame)
            self.get_logger().warn("Connectivity issue.")
            return None
        except tf2_ros.ExtrapolationException:
            print("Tried finding transformation at :", timestamp, " from ", ref_frame, " to ", target_frame)
            self.get_logger().warn("Extrapolation issue.")
            return None
    
    def get_depth_aligned_with_rgb(self, depth_img, rgb_img, cam2cam_transform = np.eye(4)):
        depth_K = (self.depth_camera_info.k[0], self.depth_camera_info.k[4], self.depth_camera_info.k[2], self.depth_camera_info.k[5])
        rgb_K = (self.rgb_camera_info.k[0], self.rgb_camera_info.k[4], self.rgb_camera_info.k[2], self.rgb_camera_info.k[5])
        return align_depth(depth_img, depth_K, rgb_img, rgb_K, cam2cam_transform)

    def pixel_to_camera(self, pixel_x, pixel_y, depth, camera_info_msg):
        """ Convert pixel coordinates to camera frame (3D point) """
        fx = camera_info_msg.k[0] if self.locobot_type > 0  else 620 # Focal length x
        fy = camera_info_msg.k[4] if self.locobot_type > 0  else 620 # Focal length y
        cx = camera_info_msg.k[2] if self.locobot_type > 0  else 320 # Principal point x
        cy = camera_info_msg.k[5] if self.locobot_type > 0  else 240 # Principal point y

        # Compute 3D position in camera frame
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth
        return np.array([x, y, z])

    def camera_to_arm_base(self, camera_coords, camera_to_arm_base):
        # Convert base_link → camera_link transform to matrix
        camera_to_arm_base_matrix = self.transform_to_matrix(camera_to_arm_base) # armbase_T_camera
        camera_coords_homogeneous = np.append(camera_coords, 1)  # Convert to (x, y, z, 1)
        arm_base_coords = camera_to_arm_base_matrix @ camera_coords_homogeneous
        return arm_base_coords[:3]  # Extract (x, y, z)
    
    def camera_to_world(self, camera_coords, base_pose, camera_to_base):
        """ Convert camera coordinates to world coordinates using /locobot/odom and TF """
        # Convert odometry pose (base_link in locobot/odom) to matrix
        base_to_odom_matrix = self.pose_to_matrix(base_pose) #world_T_base

        # Convert base_link → camera_link transform to matrix
        camera_to_base_matrix = self.transform_to_matrix(camera_to_base) #base_T_camera

        # Compute camera-to-world transformation
        camera_to_world_matrix = base_to_odom_matrix @ camera_to_base_matrix #world_T_camera

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

    # def publish_target_coord(self, coords):
    #     pose_msg = PoseStamped()
    #     pose_msg.header.stamp = self.get_clock().now().to_msg()
    #     pose_msg.pose.position.x = coords[0]
    #     pose_msg.pose.position.y = coords[1]
    #     pose_msg.pose.position.z = coords[2]

    def publish_target_coord(self, coords):
        """ Publish updated target coordinates and check if the robot is close enough """
        target_msg = TargetObject()
        target_msg.x = coords[0]
        target_msg.y = coords[1]
        target_msg.z = coords[2]

        target_msg.pose = PoseStamped()
        target_msg.pose.pose.position.x = coords[0]
        target_msg.pose.pose.position.y = coords[1]
        target_msg.pose.pose.position.z = coords[2]
        target_msg.pose.pose.orientation.x = 0.0
        target_msg.pose.pose.orientation.y = 0.0
        target_msg.pose.pose.orientation.z = 0.0
        target_msg.pose.pose.orientation.w = 1.0 # cos(theta/2)

        if self.need_world_coordinate:
        # if self.latest_odom:
            # Extract robot position from latest odometry
            robot_x = self.latest_odom.pose.pose.position.x
            robot_y = self.latest_odom.pose.pose.position.y
            robot_z = self.latest_odom.pose.pose.position.z

            # Calculate distance to target
            distance = np.sqrt((robot_x - coords[0])**2 + (robot_y - coords[1])**2 + (robot_z - coords[2])**2)


            # Set purpose = 1 if within threshold distance
            target_msg.purpose = 1 if distance <= 0.45 else 0  # Adjust threshold as needed

        else:
            target_msg.purpose = 1  # Default to 0 if no odometry data available

        self.target_coord_pub.publish(target_msg)
        self.get_logger().info(f"Published target: ({coords[0]}, {coords[1]}, {coords[2]}), Purpose: {target_msg.purpose}")

    
    def publish_debug_marker(self, coords):
        """ Publish a marker in RViz at the transformed coordinates (always expect base_link_frame) """
        marker = Marker()
        marker.header.frame_id = self.base_link_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "debug"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set marker position
        marker.pose.position.x = coords[0]
        marker.pose.position.y = coords[1]
        marker.pose.position.z = coords[2]

        marker.pose.orientation.w = 1.0  # No rotation

        # Set marker scale (size)
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        # Set color (red)
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.marker_pub.publish(marker)

    def publish_pointcloud(self, rgb_image, depth_image):
        """Generates and publishes a PointCloud2 message from RGB and depth images"""
        
        # Extract camera intrinsic parameters
        K = np.array(self.rgb_camera_info.k).reshape(3, 3)  # Camera intrinsic matrix
        fx = K[0, 0] if self.locobot_type > 0 else 620  # Focal length x
        fy = K[1, 1] if self.locobot_type > 0 else 620  # Focal length y
        cx = K[0, 2] if self.locobot_type > 0 else 320  # Principal point x
        cy = K[1, 2] if self.locobot_type > 0 else 240  # Principal point y

        height, width = depth_image.shape

        # Generate 3D point cloud
        points = []
        for v in range(height):
            for u in range(width):
                depth = depth_image[v, u] / 1000.0 if self.locobot_type > 0 else depth_image[v, u]  # Convert mm to meters if needed
                if depth == 0:  # Skip invalid depth values
                    continue
                
                # Back-project pixel (u, v) to 3D space
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth

                # Get corresponding color
                color = rgb_image[v, u]
                r, g, b = color[2], color[1], color[0]  # Convert OpenCV BGR to RGB
                rgb_packed = (r << 16) | (g << 8) | b  # Pack into uint32
                rgb_float = struct.unpack('f', struct.pack('I', rgb_packed))[0]  # Convert to float32

                points.append((x, y, z, rgb_float))

        # Convert to PointCloud2 message
        pc_msg = self.create_pointcloud2_msg(points)
        self.pc_pub.publish(pc_msg)
        self.get_logger().info(f"Published point cloud with {len(points)} points")

    def create_pointcloud2_msg(self, points):
        """Converts a list of (x, y, z, rgb) tuples into a PointCloud2 message"""
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame  # Adjust to your camera frame

        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),  # Set to FLOAT32
        ]

        msg.point_step = 20  # 4 floats (XYZ) + 1 float (RGB)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False  # Some points may have invalid depth

        # Convert point data to binary format
        buffer = []
        for x, y, z, rgb in points:
            buffer.append(struct.pack('ffff', x, y, z, rgb))  # Use 'ffff' for (x, y, z, rgb)
        msg.data = b''.join(buffer)

        return msg

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
