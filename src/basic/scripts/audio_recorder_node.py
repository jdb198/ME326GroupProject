#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sounddevice as sd
import numpy as np
import wave
from std_msgs.msg import ByteMultiArray, UInt8MultiArray

class AudioRecorderNode(Node):
    def __init__(self):
        super().__init__('audio_recorder')
        self.publisher_ = self.create_publisher(UInt8MultiArray, 'audio_data', 10)
        self.declare_parameter('record_duration', 5)  # Default: 5 sec recording
        self.record_and_publish()

    def record_and_publish(self):
        duration = self.get_parameter('record_duration').value  # Get duration from parameter
        sample_rate = 16000
        self.get_logger().info(f"Recording {duration} seconds of audio...")

        # Record audio
        recorded_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()

        # Convert numpy array to byte array
        audio_bytes = recorded_data.tobytes()

        # Publish as ROS 2 message
        msg = UInt8MultiArray() #ByteMultiArray()
        # Convert bytes to a list of integers
        # This is the key fix - properly convert bytes to a list format ROS can handle
        msg.data = [b for b in audio_bytes]  # Convert bytes to list of integers
        self.publisher_.publish(msg)

        self.get_logger().info(f"Published {len(audio_bytes)} bytes of audio.")

def main(args=None):
    rclpy.init(args=args)
    node = AudioRecorderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()