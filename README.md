# DEEP-LEARNING-PROJECT_2

**COMPANY** : CODTECH IT SOLUTIONS

**NAME** : POOJHA PRASANNA

**INTERN ID** : CT6WKMQ

**DOMAIN** : DATA SCIENCE

**BATCH DURATION** : JANUARY 10th, 2025 to FEBRUARY 25th, 2025

**MENTOR NAME** : MUZAMMIL

# **DESCRIPTION**

## **Task Description: Image Classification with CIFAR-10 Dataset**

The goal of this project is to build a deep learning model that classifies images from the CIFAR-10 dataset into one of 10 categories. This dataset consists of 60,000 32x32 color images across 10 classes, including categories such as airplanes, automobiles, birds, and cats. The task involves designing and training a Convolutional Neural Network (CNN) to classify these images, and evaluating the model's performance on a test set.

## **Tools and Libraries Used**

- **TensorFlow**:
  - A comprehensive deep learning framework used for building and training the neural network model. The Keras API within TensorFlow is utilized to define and train the CNN architecture.
  
- **NumPy**:
  - A fundamental library for numerical computing, used for handling arrays and performing operations on image data.
  
- **CIFAR-10 Dataset**:
  - A dataset containing 60,000 32x32 pixel RGB images classified into 10 categories. The dataset is divided into 50,000 training images and 10,000 test images.
  
- **Matplotlib (optional)**:
  - A visualization library used for displaying images from the CIFAR-10 dataset (if applicable for data exploration).

## **Problem Breakdown**

The CIFAR-10 dataset contains images categorized into 10 classes. The primary task is to:
- Preprocess the data (normalize and one-hot encode labels).
- Build a Convolutional Neural Network (CNN) to classify the images into these categories.
- Train the model on the training set and evaluate its performance on the test set.
- Save the trained model for future use.

## **Approach and Workflow**

1. **Data Preprocessing**:
   - The CIFAR-10 dataset is loaded using TensorFlow's `cifar10.load_data()`, and the images are normalized to values between 0 and 1. The labels are one-hot encoded to make them suitable for multi-class classification.

2. **Model Architecture**:
   - A Convolutional Neural Network (CNN) is designed with the following layers:
     - **Convolutional layers**: These layers help extract important features from the images, such as edges and textures.
     - **MaxPooling layers**: Used to down-sample the spatial dimensions, retaining only the most relevant features.
     - **Fully connected layers**: After the convolution and pooling layers, the model flattens the output and connects it to fully connected layers, which perform the classification task.
     - **Softmax layer**: The final layer outputs a probability distribution over the 10 classes, and the class with the highest probability is chosen as the predicted label.

3. **Model Training**:
   - The model is trained for 10 epochs with a batch size of 64. The Adam optimizer is used for training, and the categorical cross-entropy loss function is employed for multi-class classification.
   - The validation set (from the test data) is used to evaluate the model during training to ensure it generalizes well.

4. **Saving the Model and Predictions**:
   - After training, the model is saved to the `models/` directory as `model.h5`. This allows the trained model to be reused without retraining.
   - Predictions are made on the test set and saved as a `.npy` file in the `results/` directory.

5. **Logging**:
   - During the training process, logs are generated to capture the loss and accuracy at each epoch, which are saved in a `training_log.txt` file located in the `logs/` directory.

6. **Evaluation**:
   - After training, the model's performance is evaluated on the test set. The final accuracy is used to assess how well the model generalizes to unseen data.

## **Resources and Inspirations**
- **CIFAR-10 Dataset**:
  - A popular dataset for image classification tasks, often used to benchmark deep learning models.
  
- **TensorFlow/Keras Documentation**:
  - Comprehensive documentation providing detailed explanations and guides on using TensorFlow for building deep learning models.
  
- **Deep Learning Fundamentals**:
  - The model architecture and training process are based on standard deep learning practices, specifically using convolutional layers to process image data.

## **Conclusion**

This project demonstrates the process of building a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The approach includes data preprocessing, model architecture design, training, and evaluation. The model is saved after training and used to make predictions on unseen data. This process highlights key concepts in deep learning, such as data preprocessing, model training, and evaluation, and shows how to apply these techniques to solve real-world image classification tasks. The project also emphasizes the importance of using frameworks like TensorFlow/Keras for efficiently building and training neural networks for image classification.
