# DEEP-LEARNING-PROJECT_2

**COMPANY** : CODTECH IT SOLUTIONS

**NAME** : POOJHA PRASANNA

**INTERN ID** : CT6WKMQ

**DOMAIN** : DATA SCIENCE

**BATCH DURATION** : JANUARY 10th, 2025 to FEBRUARY 25th, 2025

**MENTOR NAME** : MUZAMMIL

# DESCRIPTION

Task Description: Image Classification using CIFAR-10 Dataset
The goal of this project is to build a deep learning model to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset contains 60,000 32x32 color images across 10 classes, with each class representing a different type of object. This project focuses on training a Convolutional Neural Network (CNN) to classify these images into one of the 10 classes, achieving high accuracy through model training and evaluation.

Tools and Libraries Used
TensorFlow:
A powerful deep learning framework used to define, train, and evaluate the neural network model. TensorFlow's Keras API is used for building and training the CNN.
NumPy:
A fundamental package for numerical computing in Python, used for data manipulation and saving model predictions.
CIFAR-10 Dataset:
The dataset used for training and testing the model. It contains 60,000 32x32 RGB images divided into 10 classes.
Matplotlib (optional):
A library used to visualize the images from the dataset (if required for visualization).
Problem Breakdown
The CIFAR-10 dataset consists of 60,000 images divided into 10 classes, such as airplanes, automobiles, birds, cats, and more. The objective is to build a convolutional neural network (CNN) that:

Accepts the 32x32 RGB images as input.
Classifies the images into one of the 10 predefined categories.
Achieves good performance on the test set, typically by improving accuracy over time during training.
Approach and Workflow
Data Preprocessing:

The CIFAR-10 dataset is loaded and normalized to have pixel values between 0 and 1.
Labels are one-hot encoded to prepare them for training.
Model Architecture:

A simple CNN is designed with the following layers:
Convolutional layers: Extract important features from the image (e.g., edges, textures).
Max-pooling layers: Reduce spatial dimensions, retaining only the most important information.
Fully connected layers: Flatten the output from the convolutional layers and perform the final classification into one of 10 classes using a softmax output layer.
Model Training:

The model is trained using the training data for 10 epochs, with a batch size of 64.
The Adam optimizer and categorical crossentropy loss function are used for training.
Validation accuracy is monitored using the test data.
Model Saving and Predictions:

After training, the model is saved in the models/ directory as model.h5.
Predictions are made on the test set and saved as a .npy file in the results/ directory.
Logging:

During training, the loss and accuracy of each epoch are logged into training_log.txt in the logs/ directory for later analysis.
Evaluation:

After training, the model's performance is evaluated using the test set to assess its accuracy on unseen data.
Resources and Inspirations
CIFAR-10 Dataset: A widely used dataset for image classification tasks, often used for benchmarking image classification models.
TensorFlow/Keras Documentation: Provides a comprehensive guide on how to build, train, and evaluate deep learning models.
Deep Learning Concepts: The approach and model architecture used are based on fundamental deep learning principles, particularly convolutional neural networks (CNNs) for image classification.
Conclusion
This deep learning project demonstrates the process of building and training a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. The approach involves data preprocessing, building the model, training, and evaluating its performance. The model is saved and evaluated for further analysis. This project not only highlights the importance of data preprocessing and model architecture selection but also provides insight into the workflow of training deep learning models for image classification tasks. By utilizing tools like TensorFlow and Keras, the project showcases how powerful deep learning frameworks can simplify the process of developing sophisticated models for real-world tasks such as image recognition.



