import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import pickle  # Import pickle module for loading metadata
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

labels_to_include = ['Drag', 'Loupe', 'Point', 'Scale', 'None']

# Define the VGG-16 based model
class VGG16Model(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Model, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        # Replace the last fully connected layer
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)
    
    def forward(self, x):
        return self.vgg(x)

# Load the VGG-16 model
model = VGG16Model(len(labels_to_include)).to('cuda')
model.load_state_dict(torch.load('test_model/hand_gesture_model_VGG16.pth'))  # Load model state dictionary

# Load the metadata
with open('test_model/VGG-16_metadata.pkl', 'rb') as f:
    model_metadata = pickle.load(f)

# Set the model to evaluation mode
model.eval()

# Function to perform inference on a single image
def predict_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply transformations
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to('cuda')  # Move to GPU if available

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    
    # Map predicted index to label
    predicted_label = model_metadata['class_labels'][predicted.item()]

    return predicted_label

# Example usage: Capture from laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction on the frame
    predicted_label = predict_image(frame)

    # Display predicted label on the frame
    cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition with VGG-16 model', frame)

    # Break the loop on pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
