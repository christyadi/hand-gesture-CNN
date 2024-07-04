# Hand Gesture Recognition

By Jonathan Christyadi - 502705


https://github.com/christyadi/hand-gesture-CNN/assets/118226145/04a0cfd8-d078-461e-a683-e36f479d9f57

https://github.com/christyadi/hand-gesture-CNN/assets/118226145/c42afbcb-6a29-4229-8d8f-d3da5a4ae197

## Introduction

Hand gesture recognition is a critical component of human-computer interaction, enabling intuitive control of devices using natural hand movements. This project focuses on the development and evaluation of a Convolutional Neural Network (CNN) and VGG-16 model for recognizing various hand gestures, utilizing a comprehensive dataset provided by the Universidad de Alicante. The dataset comprises real images, synthetic images, and scene descriptions, designed to train and evaluate hand gesture detection methods.

## Dataset Overview

The original images and videos were recorded at a resolution of 1,920 x 1,080 pixels. However, for performance and accuracy reasons, the images were scaled down to a spatial resolution of 224 x 224 pixels.

- **Dataset sources**: [Universidad de Alicante Gesture Dataset](https://www.dlsi.ua.es/~jgallego/datasets/gestures/)

### Gesture Categories and Dataset Summary

The dataset encompasses the following gesture categories:

- **Point**: Images of pointing gestures.
- **Drag**: Images including drag gestures.
- **Loupe**: Samples including loupe gestures.
- **Pinch**: Sequences of dynamic pinch gestures.
- **None**: Samples where no hand appears.
  
![hand_gesture_samples](https://github.com/christyadi/hand-gesture-CNN/assets/118226145/52100c0d-e622-4524-950f-ef8e44422902)

The dataset is highly suitable for training robust hand gesture recognition models due to its quality and diversity, as evidenced by several key factors:

- **Variety in Real Images**: The real images dataset includes frames extracted from videos recorded in varied environments, including indoor and outdoor scenes, with different lighting conditions and backgrounds.
- **Balanced Gesture Representation**: The dataset ensures balanced representation of each gesture category, with a comparable number of samples for point, drag, loupe, pinch, and none.
- **Comprehensive Annotations**: The dataset provides detailed annotations, including bounding boxes for hand positions, fingertips, and objects pointed to.
- **High-Quality Image Resolution**: Although the original images were recorded at a high resolution of 1,920 x 1,080 pixels, they were downscaled to 224 x 224 pixels for practical reasons.

**Note**: One limitation of the dataset is the skin tone color of the hand in the pictures. Most of the skin color in the images are either white or brown but does not include black.

## Project Objectives

This project aims to build a robust CNN model capable of accurately recognizing hand gestures from images, leveraging the rich and varied dataset provided. The model will be trained, validated, and tested on real data to ensure generalization across different environments and conditions. Additionally, the project includes a data preprocessing pipeline, data augmentation strategies, and a detailed evaluation process to measure the model's performance.

## Evaluation Metrics

The model's performance is evaluated using the following metrics:

- **Confusion Matrix**: A table used to describe the performance of the classification model.
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.
- **Loss and Accuracy Curves**: Visualizes the training and validation loss and accuracy over epochs.



