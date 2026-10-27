# Vegetable Image Classification Using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) from scratch to classify vegetable images into 15 different classes.

The model was trained using TensorFlow/Keras on Kaggle and deployed using Streamlit to allow users to upload a vegetable image and receive a predicted class with a confidence score.

## Dataset

The dataset contains vegetable images divided into three subsets:

- Training: 15,000 images
- Validation: 3,000 images
- Testing: 3,000 images
- Number of classes: 15

The dataset is organized into separate folders for training, validation, and testing.

## Preprocessing

Images were prepared using `ImageDataGenerator`.

The preprocessing steps included:

- Resizing images to 128 × 128 pixels
- Normalizing pixel values by dividing by 255
- Using categorical class labels
- Applying horizontal flipping as data augmentation to the training set only

## CNN Model

A custom CNN architecture was created from scratch without using pretrained models.

The network includes:

- Conv2D layers
- MaxPooling2D layers
- Flatten layer
- Dense layer
- Softmax output layer for 15 classes

The model was compiled using:

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metric: Accuracy

## Training

The model was trained for exactly 2 epochs as required.

Training results:

### Epoch 1
- Training Accuracy: 58.68%
- Validation Accuracy: 77.47%
- Training Loss: 1.2452
- Validation Loss: 0.7109

### Epoch 2
- Training Accuracy: 84.92%
- Validation Accuracy: 84.30%
- Training Loss: 0.4763
- Validation Loss: 0.4963

## Callbacks

The following callbacks were used during training:

- EarlyStopping
- ModelCheckpoint

The best model was saved during training.

## Evaluation

The trained model was evaluated on the test dataset using accuracy and loss.

Training and validation accuracy/loss curves were also plotted.

Predictions were generated for test images and compared with their actual classes.

## Two-Epoch Training Limitation

The model was intentionally trained for only two epochs according to the assignment requirements.

Although the model achieved good validation accuracy, additional epochs could potentially improve convergence and classification performance.

## Saved Files

The trained model was saved as:

`vegetable_cnn.keras`

The class names were saved as:

`class_names.json