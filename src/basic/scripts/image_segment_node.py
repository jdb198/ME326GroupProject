#!/usr/bin/env python3

import torch
import numpy as np 
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import ByteMultiArray, UInt8MultiArray, String, Float32MultiArray
from PIL import Image
import torch
import cv2
from ultralytics import YOLO
from open_clip import create_model, tokenize  # Assuming these are defined elsewhere
import os

class ImageSegmentNode(Node):
    def __init__(self):
        super().__init__('image_segment')
        
        # Create publisher for center coordinates
        self.publisher = self.create_publisher(
            Float32MultiArray, 
            'object_center', 
            10
        )
        
        # Subscribe to transcribed text
        self.text_subscription = self.create_subscription(
            String,
            'transcribed_text',
            self.text_callback,
            10
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info("Image segment node started. Waiting for transcribed text...")
        
        # Placeholder for the image path - in a real application, this might come from another node
        self.image_path = "../../../saved_images/image_2.png"
        
    def text_callback(self, msg):
        """Process the received transcribed text and use it as a prompt for object detection"""
        text_prompt = msg.data
        self.get_logger().info(f"Received text prompt: {text_prompt}")
        
        try:
            # Process the image using the received text prompt
            center_x, center_y = self.get_center(self.image_path, text_prompt)
            
            # Publish the center coordinates
            center_msg = Float32MultiArray()
            center_msg.data = [float(center_x), float(center_y)]
            self.publisher.publish(center_msg)
            
            self.get_logger().info(f"Published object center: ({center_x}, {center_y})")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def preprocess_pil_image(self, image):
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return image_tensor

    def get_center(self, image_path, text_prompt):
        """Find the center of the object in the image that best matches the text prompt"""
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Load YOLOv11 model (pre-trained on COCO dataset)
        model = YOLO("yolo11n.pt")  # YOLOv11 model
        results = model(image_path)  # Perform inference

        # Extract detection results from the tensor
        detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data as NumPy array
        boxes = detections[:, :4]  # Bounding box coordinates
        scores = detections[:, 4]  # Confidence scores
        classes = detections[:, 5].astype(int)  # Class indices

        # Access class names from the YOLOv11 model
        class_names = model.names  # Mapping class indices to names

        # Ensure that boxes exist
        if len(boxes) == 0:
            raise ValueError("No objects detected in the image!")

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

        for i, box in enumerate(boxes):
            # Crop object from the image
            xmin, ymin, xmax, ymax = map(int, box)  # Ensure indices are integers
            # cropped_object = image.crop((xmin, ymin, xmax, ymax))
            cropped_object = image[ymin:ymax, xmin:xmax]
            # convert back to pil
            cropped_object = Image.fromarray(cropped_object)

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


def main(args=None):
    rclpy.init(args=args)
    node = ImageSegmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()