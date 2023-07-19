# MNIST Handwritten Digits Recognition with Keras

This repository contains a simple handwritten digits recognition project using Keras. The model is trained to classify images of handwritten digits from the famous MNIST dataset.

## Project Overview

The MNIST dataset is a popular benchmark dataset in the field of machine learning. It consists of 28x28 grayscale images of handwritten digits (0 to 9) along with their corresponding labels. The goal of this project is to train a deep learning model that can accurately classify these digits.

## Prerequisites

Before running the code, make sure you have the following installed:

- Python (>=3.9)
- Keras (>=2.6.0)
- TensorFlow (>=2.6.0)
- Numpy (>=1.19.5)
- Matplotlib (>=3.4.3)

You can install the required packages using the following command:


## Load Data

The Amnesty dataset is imported using the line: `from tensorflow.keras.datasets import mnist`. The dataset provides training data, training labels, test data, and test labels as separate variables.

It is recommended to check if a GPU is available for use. A function using TensorFlow's device library can be used to list available devices, including GPUs.

The downloaded dataset contains 50,000 training images and 10,000 test images. If a GPU is not enabled, it can be changed in the notebook settings by switching to hardware acceleration.



## View and Inspect Data

Inspecting the dataset is an essential step to understand and explore the data. The code provided prints the shape and length of the training and test datasets, as well as the dimensions of one image sample and the shape of all the training labels.

The MNIST dataset contains 60,000 training images, each with dimensions of 28x28 pixels in grayscale. The dataset is stored in variables named "x_train," "y_train," "x_test," and "y_test" after being loaded.


## Visualizing the Data

Visualizing the data is important for a sanity check and to gain a better understanding of the dataset. In Keras, visualizing the data is easier as it does not require converting the data into an ideal format.

Random samples can be accessed using the random function and the length of the training dataset. The samples can be visualized using the Matplotlib function and OpenCV to convert the images from BGR to RGB. The ground truth labels can be displayed in the title of each image. Multiple images can be plotted using subplots and the cell block function.



## Preprocess Data

Pre-processing the data is necessary to meet the requirements of Keras models. Keras expects data in a specific format and may require additional modifications for proper processing.

The following pre-processing steps are performed:

1. Adding an extra dimension to the data.
2. Converting the data type from unsigned integer to float32.
3. Normalizing the data between 0 and 1.
4. Performing one-hot encoding for the labels.
5. Reshaping the data to the desired shape using the reshape function.

The pre-processed data is ready to be used for training the model.


## Building CNN

To build the CNN model, Keras provides convenient building blocks for constructing the model layers. The model architecture used in this project is a simple feedforward neural network with convolutional and fully connected layers.



## Training the Model

Training the CNN model in Keras is relatively simple using the model.fit() function. After compiling the model, training can be initiated using the model.fit() function. The model.fit() function takes the training data, training labels, batch size, number of epochs, and other parameters.



## Plotting the Results

To visualize the training progress, the history.history dictionary can be accessed after training. The history.history dictionary stores the results of metrics such as accuracy, loss, validation accuracy, and validation loss.



## Saving and Loading the Model

To save a trained model in Keras, the `model.save()` function can be used, specifying the desired name for the saved model. The saved model will be in the form of an HDF5 file with a .h5 extension.


## Additional Functionality

The provided code includes an example of using OpenCV to create images displaying the original test image along with the predicted class label. This code randomly selects 10 images from the test dataset, resizes them, reshapes them, and generates images with the predicted class labels displayed.

Feel free to explore the code further and make any modifications as needed for your specific use case. Happy coding!
