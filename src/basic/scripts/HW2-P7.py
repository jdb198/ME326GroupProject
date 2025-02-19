#!/usr/bin/env python3

# Import Libraries
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, Point
from nav_msgs.msg import Odometry
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from rclpy.serialization import serialize_message
import rosbag2_py
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker

class LocobotBaseMotionTracking(Node):
    def __init__(self):
        super().__init__('locobot_base_motion_tracking')

        # Define publisher and subscribers
        self.velocity_publisher = self.create_publisher(Twist, '/locobot/diffdrive_controller/cmd_vel_unstamped', 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.t_init = self.get_clock().now()  # Define the initial time

        # Initialize the x and y coordinate calculation and measurement variables
        self.x_current = 0
        self.y_current = 0
        self.x_odom = 0
        self.y_odom = 0

        self.target_pose = Pose()
        self.target_pose.position.x = 2.0
        self.target_pose.position.y = -2.0

        self.target_pose.orientation.x = 0.0
        self.target_pose.orientation.y = 0.0
        self.target_pose.orientation.z = 0.0
        self.target_pose.orientation.w = 1.0 # cos(theta/2)

        # This is the distance of the point P (x,y) that will be controlled for position. The locobot base_link frame points 
        # forward in the positive x direction, the point P will be on the positive x-axis in the body-fixed frame of the robot 
        # mobile base
        self.L = 0.1

        # Define Kp and Ki variables
        self.Kp = 1.2
        self.Ki = 0.2

        self.integrated_error = np.matrix([[0],[0]]) #this is the integrated error for Proportional, Integral (PI) control
        # self.integrated_error_factor = 1.0 #multiply this by accumulated error, this is the Ki (integrated error) gain
        self.integrated_error_list = []
        self.length_of_integrated_error_list = 20
        self.goal_reached_error = 0.05

        # # Initialize data array and prepare the CSV and bag files for export
        # self.data = [] 
        # self.csv_file_path = os.path.expanduser('~/bags/locobot_odom_data.csv')
        # self.bag_file_path = os.path.expanduser('~/bags/locobot_odom')
        # os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)

        # # Define the rosbag2 variables and run the bag writer finctions
        # self.writer = rosbag2_py.SequentialWriter()
        # storage_options = rosbag2_py.StorageOptions(uri=self.bag_file_path, storage_id='sqlite3')
        # converter_options = rosbag2_py.ConverterOptions(input_serialization_format='', output_serialization_format='')
        # self.writer.open(storage_options, converter_options)
        # self.writer.create_topic(rosbag2_py.TopicMetadata(name='/locobot/recorded_odom', type='nav_msgs/msg/Odometry', serialization_format='cdr'))

        self.get_logger().info('The velocity_publisher node has started.')  # Relay node start message to user

    # Function to calculate the x and y position of the locobot at a given time
    # def calc_traj(self, t):
    #     x = 0.5 * np.cos((2 * np.pi * t) / 10)
    #     y = 0.5 * np.sin((2 * np.pi * t) / 10)

    #     return x, y

    def pub_point_P_marker(self):
        #this is very simple because we are just putting the point P in the base_link frame (it is static in this frame)
        marker = Marker()
        marker.header.frame_id = "locobot/base_link"
        marker.header.stamp = self.get_clock().now().to_msg() #used to be in ROS1: rospy.Time.now()
        marker.id = 0
        marker.type = Marker.SPHERE
        # Set the marker scale
        marker.scale.x = 0.1  # radius of the sphere
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Set the marker pose
        marker.pose.position.x = self.L  # center of the sphere in base_link frame
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        # Set the marker color
        marker.color.a = 1.0 #transparency
        marker.color.r = 1.0 #red
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Publish the marker
        self.point_P_control_point_visual.publish(marker)

    def pub_target_point_marker(self):
        #this is putting the marker in the world frame (http://wiki.ros.org/rviz/DisplayTypes/Marker#Points_.28POINTS.3D8.29)
        marker = Marker()
        marker.header.frame_id = "locobot/odom" #this will be the world frame for the real robot
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.ARROW
        # Set the marker scale
        marker.scale.x = 0.3  # arrow length
        marker.scale.y = 0.1 #arrow width
        marker.scale.z = 0.1 #arrow height
        

        # Set the marker pose
        marker.pose.position.x = self.target_pose.position.x  # center of the sphere in base_link frame
        marker.pose.position.y = self.target_pose.position.y
        marker.pose.position.z = self.target_pose.position.z
        marker.pose.orientation.x = self.target_pose.orientation.x
        marker.pose.orientation.y = self.target_pose.orientation.y
        marker.pose.orientation.z = self.target_pose.orientation.z
        marker.pose.orientation.w = self.target_pose.orientation.w

        # Set the marker color
        marker.color.a = 1.0 #transparency
        marker.color.r = 0.0 #red
        marker.color.g = 1.0
        marker.color.b = 0.0

        # Publish the marker
        self.target_pose_visual.publish(marker)

    # Odometry callback function
    def odom_callback(self, odom_msg: Odometry):
        # Obtain the x and y measured values from the odometry
        self.x_odom = odom_msg.pose.pose.position.x
        self.y_odom = odom_msg.pose.pose.position.y

        # Convert the odometry readings to quaternion coordinates
        qw = odom_msg.pose.pose.orientation.w
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z

        # Generate the quaternions into rotation matrix entries
        R11 = qw**2 + qx**2 - qy**2 - qz**2
        R12 = 2*qx*qz + 2*qw*qz
        R21 = 2*qx*qz - 2*qw*qz
        R22 = qw**2 - qx**2 + qy**2 - qz**2

        # t = (self.get_clock().now() - self.t_init).nanoseconds * 1e-9  # Get elapsed time since the initial time

        # # Edit the Kp value if one full rotation has passed
        # if t > 20:
        #     self.get_logger().info('It has been 20 seconds.')
        #     self.destroy_node()
        #     return
        
        # Get desired position
        # self.x_current, self.y_current = self.calc_traj(t)
        point_P = Point()
        #NOTE: the following assumes that when at the origin, the baselink and odom/world frame are aligned, and the z-axis points up. If this is not true then there is not a simple rotation about the z-axis as shown below
        point_P.x = self.x_odom + self.L*R11
        point_P.y = self.y_odom + self.L*R21
        point_P.z = 0.1 #make it hover just above the ground (10cm)
        
        # Compute errors and put them in a vector
        err_x = self.target_pose.position.x - point_P.x
        err_y = self.target_pose.position.y - point_P.y
        error_vect = np.matrix([[err_x],[err_y]])

        Kp_mat = self.Kp * np.eye(2)  # Make the Kp identity matrix
        Ki_mat = self.Ki * np.eye(2)  # Make the Ki identity matrix

        Rotation_mat = np.matrix([[R11,R12],[R21,R22]])  # Define rotation matrix

        current_angle = np.arctan2(Rotation_mat[0,1], Rotation_mat[1,1])  # Find the current angle

        self.integrated_error_list.append(error_vect)
        if len(self.integrated_error_list) > self.length_of_integrated_error_list:
            self.integrated_error_list.pop(0) #remove last element
        #now sum them
        self.integrated_error = np.matrix([[0],[0]])
        for err in self.integrated_error_list:
            self.integrated_error = self.integrated_error + err

        point_p_error_signal = Kp_mat * error_vect  + Ki_mat * self.integrated_error # Find the point error signal

        # Define non-holonomic matrix and find the control input matrix
        non_holonomic_mat = np.matrix([[np.cos(current_angle), -self.L*np.sin(current_angle)], [np.sin(current_angle), self.L * np.cos(current_angle)]])
        control_input = np.linalg.inv(non_holonomic_mat) * point_p_error_signal

        # Proportional control: Finding the velocity and angular velocity
        v = control_input.item(0)  # Velocity proportional to distance error
        omega = control_input.item(1)  # Angular velocity

        err_magnitude = np.linalg.norm(error_vect)
        # net_error_magnitude = np.linalg.norm(point_p_error_signal)
        
        # Publish velocity message
        control_msg = Twist()
        control_msg.linear.x = float(v)
        control_msg.angular.z = float(omega)

        if np.linalg.norm(control_input) > 2:
            control_msg.linear.x = control_msg.linear.x/np.linalg.norm(control_input)
            control_msg.angular.z = control_msg.angular.z/np.linalg.norm(control_input)

        self.velocity_publisher.publish(control_msg)

        if err_magnitude < self.goal_reached_error:
            return

        # # Append the odometry and time data to the data array
        # self.data.append([t, self.x_odom, self.y_odom])

        # # Send the odometry and time data to the bag file
        # odom_serialized = serialize_message(odom_msg)
        # self.writer.write('/locobot/recorded_odom', odom_serialized, self.get_clock().now().nanoseconds)

        # Report the data to the user for inspection
        self.get_logger().info(f'Odometry: ({self.x_odom:.2f}, {self.y_odom:.2f}), Target: ({self.x_current:.2f}, {self.y_current:.2f}), Error: ({err_x:.2f}, {err_y:.2f}), Kp: {self.Kp}')

    # # Function to export the CSV file
    # def export_csv(self):
    #     with open(self.csv_file_path, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['Time', 'X_Val', 'Y_Val'])
    #         writer.writerows(self.data)
    #     self.get_logger().info(f'Data saved to {self.csv_file_path}')

# Main function to control the node
def main(args=None):
    rclpy.init(args=args)
    node = LocobotBaseMotionTracking()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        # node.save_data_to_csv()
        node.destroy_node()
        rclpy.shutdown()

# Run main function
if __name__ == '__main__':
    main()
