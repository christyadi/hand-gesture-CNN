import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Constants
IMAGE_SIZE = (224,224)
MODEL_PATH = 'saved_models/hand_gesture_cnn.pth'
LABELS = ['Drag', 'Loupe', 'None', 'Point', 'Scale']
# LABELS = ['Drag', 'Loupe', 'Other', 'Point', 'Scale', 'Fingertip']

# Function to capture an image using webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Image (Press "s" to save and "q" to quit)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save the captured image
            img_name = "captured_image.png"
            cv2.imwrite(img_name, frame)
            print(f"Image saved as {img_name}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return img_name

# Function to preprocess the image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function to load the model and predict the gesture
def predict_gesture(img_path):
    model = load_model(MODEL_PATH)
    image = preprocess_image(img_path)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction[0])
    class_label = LABELS[class_idx]
    confidence = prediction[0][class_idx]
    return class_label, confidence

if __name__ == "__main__":
    # Capture image
    captured_image_path = capture_image()
    
    # Predict gesture
    label, conf = predict_gesture(captured_image_path)
    
    # Output the result
    print(f"Predicted Gesture: {label} (Confidence: {conf * 100:.2f}%)")
