# Fake-image-detection

## Introduction
Deep fake images have become a growing concern in today's digital age. This project aims to develop a machine learning model to accurately classify real and fake images using convolutional neural networks (CNNs). The goal is to detect manipulations in images and distinguish between authentic and altered content.

## Features
- Binary Classification: Classifies images as either real or fake.
- Pre-trained Model: Utilizes the VGG16 model to improve performance.
- Transfer Learning: Adds custom layers to fine-tune the pre-trained model for the dataset.

## Dataset
The dataset consists of images stored in two categories:

*Real*: Images that are authentic.
*Fake*: Images that are generated/manipulated to create deep fakes.
The dataset is split into training, validation, and testing sets for model evaluation.

## Model Architecture
The model uses the VGG16 architecture with added layers for classification:

*Input Layer*: Pre-trained VGG16 layers.
*Custom Layers*: Additional Dense, Dropout, and Activation layers for fine-tuning.
*Output Layer*: Sigmoid activation function for binary classification.

## Results
*Accuracy*: Achieved an accuracy of around 85% using the VGG16 model after fine-tuning.
*Loss*: The final model showed minimal loss on the validation set.

## Technologies Used
*Python*: Core programming language.
*TensorFlow/Keras*: For building and training the deep learning model.

## Future Work
*Improve Model Accuracy*: Experiment with other pre-trained models (ResNet, EfficientNet) to boost performance.
*More Azure Services*: Utilize Azure Face API and AI Language Services to add further layers of detection and analysis.
*Real-time Deepfake Detection*: Implement real-time detection of deep fakes in video streams.
