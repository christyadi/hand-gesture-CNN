import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

labels_to_include = ['Drag', 'Loupe', 'Point', 'Scale', 'None']

# Define a more complex CNN model with batch normalization and dropout
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, len(labels_to_include))
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Load the model architecture
model = CNNModel()

# Load the model state dictionary
model_metadata = torch.load('test_model\hand_gesture_model_40val_metadata.pth')
model.load_state_dict(model_metadata['model_state_dict'])

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

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    
    # Map predicted index to label
    predicted_label = model_metadata['labels_to_include'][predicted.item()]

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
    cv2.imshow('Hand Gesture Recognition, press Q to exit', frame)

    # Break the loop on pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
