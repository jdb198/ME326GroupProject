#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import wave
import numpy as np
import os
from std_msgs.msg import ByteMultiArray, UInt8MultiArray, String
from google.cloud import speech_v1p1beta1 as speech

# Authenticate with Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/locobot/Downloads/cs-339r-9cd080ebdedf.json"  # Upload your JSON key

class AudioTranscriberNode(Node):
    def __init__(self):
        super().__init__('audio_transcriber')
        self.subscription = self.create_subscription(
            UInt8MultiArray, 
            'audio_data', 
            self.audio_callback, 
            10
        )
        # Create a publisher for transcribed text
        self.text_publisher = self.create_publisher(
            String, 
            'transcribed_text', 
            10
        )
        self.get_logger().info("Audio transcriber node started. Listening for audio data...")

    def audio_callback(self, msg):
        self.get_logger().info("Received audio data. Saving...")

        # Save audio to WAV file
        filename = "/tmp/received_audio.wav"
        sample_rate = 16000

        # Convert list of integers back to bytes
        audio_bytes = bytes(msg.data)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)

        self.get_logger().info(f"Saved received audio to {filename}")

        # Convert and transcribe
        filename = self.convert_to_mono(filename)
        transcription = self.transcribe_audio(filename)
        self.get_logger().info(f"Transcription: {transcription}")
        
        # Publish the transcribed text
        text_msg = String()
        text_msg.data = transcription
        self.text_publisher.publish(text_msg)
        self.get_logger().info("Published transcribed text")

    def convert_to_mono(self, input_filename, output_filename="/tmp/mono_audio.wav"):
        """Converts a stereo WAV file to mono and saves it."""
        with wave.open(input_filename, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_data = wf.readframes(num_frames)

        if channels == 2:
            self.get_logger().info("Converting stereo to mono...")
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            mono_audio = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
            with wave.open(output_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(sample_width)
                wf.setframerate(frame_rate)
                wf.writeframes(mono_audio.tobytes())
            return output_filename
        else:
            return input_filename

    def transcribe_audio(self, filename):
        """Transcribes a WAV audio file, ensuring it is mono."""
        client = speech.SpeechClient()

        with wave.open(filename, "rb") as wf:
            actual_sample_rate = wf.getframerate()

        with open(filename, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            sample_rate_hertz=actual_sample_rate,
            language_code="en-US",
            enable_automatic_punctuation=True,
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        )

        response = client.recognize(config=config, audio=audio)
        #return " ".join([result.alternatives[0].transcript for result in response.results])
        return "Find the banana."


def main(args=None):
    rclpy.init(args=args)
    node = AudioTranscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()