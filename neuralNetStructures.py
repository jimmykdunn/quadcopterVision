# -*- coding: utf-8 -*-
"""
FILE: neuralNetStructures.py
DESCRIPTION:
    Function for the forward passes of hourglass neural networks.
    Uses tensorflow.

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""

import tensorflow as tf
import nnUtilities as nnu

"""
Build the hourglass NN template.  This is a 4-hidden layer convolutional neural
network. It has 2 layers of convolution+pooling, followed by two layers of 
upconvolution, including a skip connection.  Set up as a tensorflow graph so
that we can train its weight (w) and bias (b) parameters.

INPUTS:
    x: input images, [nBatch,width,height,1]
EXAMPLE:
    heatmap = hourglass_nn(x)
RETURNS:
    heatmaps: map of pixel values, higher for pixles more likely to be target.
        Same size as input x.
"""
def hourglass_nn(x):
    
    nBatch, nWidth, nHeight = x.shape[:3]
    
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x,[-1,nWidth,nHeight,1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('firstConv'):
        w1 = nnu.weight_variable([5,5,1,32])
        b1 = nnu.weight_variable([32])
        #w1 = weight_variable([7,7,1,32])
        #h_conv1 = tf.nn.relu(nnu.conv2d(x_image,w1)) # [-1,width,height,32]
        h_conv1 = tf.nn.relu(tf.nn.bias_add(nnu.conv2d(x_image,w1), b1)) # added bias term

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('firstPool'):
        h_pool1 = nnu.max_pool(h_conv1,4) # [-1,width/4,height/4,32]

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('secondConv'):
        w2 = nnu.weight_variable([5,5,32,64]) 
        b2 = nnu.weight_variable([64])
        #w2 = weight_variable([7,7,32,64]) 
        #h_conv2 = tf.nn.relu(nnu.conv2d(h_pool1,w2)) # [-1,width/4,height/4,64]
        h_conv2 = tf.nn.relu(tf.nn.bias_add(nnu.conv2d(h_pool1,w2),b2))

    # Second pooling layer.
    with tf.name_scope('secondPool'):
        h_pool2 = nnu.max_pool(h_conv2,2) # [-1,width/8,height/8,64]  

    # Remember the order is skip-connection THEN upconv  

    with tf.name_scope('secondUpconv'):
        # No skip connection necessary on the innermost layer
        h_upconv1 = tf.nn.relu(nnu.upconv2d(h_pool2, [3,3,32,64])) # [-1,width/4,height/4,32]
        
    with tf.name_scope('firstUpconv'):
        h_sk1 = nnu.addSkipConnection(h_upconv1, h_pool1) # skip connection [-1,width/4,height/4,64]
        heatmaps = tf.nn.relu(nnu.upconv2d(h_sk1, [5,5,1,64])) # [-1,width,height,1]
        
    # The size of heatmap here should be the same as the original images
    return heatmaps
    
# end hourglass_nn

