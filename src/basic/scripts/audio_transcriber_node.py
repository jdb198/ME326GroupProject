import rclpy
from rclpy.node import Node
import wave
import numpy as np
import os
from std_msgs.msg import ByteMultiArray
from google.cloud import speech_v1p1beta1 as speech

class AudioTranscriberNode(Node):
    def __init__(self):
        super().__init__('audio_transcriber')
        self.subscription = self.create_subscription(ByteMultiArray, 'audio_data', self.audio_callback, 10)
        self.get_logger().info("Audio transcriber node started. Listening for audio data...")

    def audio_callback(self, msg):
        self.get_logger().info("Received audio data. Saving...")

        # Save audio to WAV file
        filename = "/tmp/received_audio.wav"
        sample_rate = 16000

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(msg.data))

        self.get_logger().info(f"Saved received audio to {filename}")

        # Transcribe using Google Cloud
        transcription = self.transcribe_audio(filename)
        self.get_logger().info(f"Transcription: {transcription}")

    def transcribe_audio(self, filename):
        """Uses Google Cloud Speech-to-Text to transcribe a WAV file."""
        client = speech.SpeechClient()

        with open(filename, 'rb') as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            sample_rate_hertz=16000,
            language_code='en-US',
            enable_automatic_punctuation=True,
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        )

        response = client.recognize(config=config, audio=audio)
        return " ".join([result.alternatives[0].transcript for result in response.results])


def main(args=None):
    rclpy.init(args=args)
    node = AudioTranscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
