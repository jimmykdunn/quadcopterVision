# -*- coding: utf-8 -*-
"""
FILE: nmist.py
DESCRIPTION:
Loads the NMIST handwritten-digits data using the built-in keras function.
Performs normalization and also displays a few example images if python 
terminal allows it.

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
getNMISTData
INPUTS:
    None
EXAMPLE:
    x_train, y_train, x_test, y_test = getMNISTData()
RETURNS:
    x_train: full set of 60000 28x28 pixel training images, [60000,28,28]
    y_train: full set of 60000 truth classes for x_train [60000]
    x_test: full set of 10000 28x28 pixel test images, [10000,28,28]
    y_test: full set of 10000 truth classes for x_test [10000]
"""
def getMNISTData():
    """Load training and eval data"""
    ((x_train, y_train),
     (x_test, y_test)) = tf.keras.datasets.mnist.load_data()
    print("Number of training images: " + str(x_train.shape[0]))
    print("Number of testing  images: " + str(x_test.shape[0]))
    print("Image shape (nx,ny): (" + str(x_train.shape[1]) + 
                               "," + str(x_train.shape[2]) + ")")
    
    x_train = x_train/np.float32(255) # normalize images
    y_train = y_train.astype(np.int32)  # not required

    x_test = x_test/np.float32(255) # normalize images
    y_test = y_test.astype(np.int32)  # not required
    
    # Display some example data
    chain = np.squeeze(x_train[0,:,:])
    for i in range(15):
        chain = np.append(chain,np.squeeze(x_train[i+1,:,:]),axis=1)
    print("Example training data")
    plt.figure()
    plt.imshow(chain)
    labelstrs = "".join([str(truth) + ", " for truth in y_train[:16]])
    print("Truth: ", labelstrs)
    
    return x_train, y_train, x_test, y_test
