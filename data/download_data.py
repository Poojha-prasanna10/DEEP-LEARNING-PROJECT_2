import os
import tensorflow as tf
import numpy as np

def download_data():
    # Create 'data/' folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Save the dataset in the 'data/' folder
    np.savez('data/cifar10_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print("Dataset downloaded and saved in 'data/cifar10_data.npz'.")

if __name__ == '__main__':
    download_data()
