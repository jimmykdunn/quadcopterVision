# -*- coding: utf-8 -*-
"""
FUNCTION: run_cnn
DESCRIPTION:
    Trains and evaluates a convolutional neural net (CNN) to generate a heatmap
    of the probability of a quadcopter being at each pixel in the input image.
    Uses tensorflow.  
    
    Based on NMIST tutorial at https://www.tensorflow.org/tutorials/estimators/cnn
    
INPUTS: 

    
OUTPUTS: 
    

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

tf.logging.set_verbosity(tf.logging.INFO)

# This function defines the forward pass of the network. The backward pass is
# executed implicitly by tensorflow.
# The default implementation has 2 hidden conv+maxpool layers followed by a 
# single fully-connected (dense) layer.
# INPUTS:
#     features: The data itself.  [nImages,nx,ny,nColors]. Dictonary for some reason
#     labels: True classes of each image in features
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input layer.
    # Note that features["x"] just extracts that part of the dictionary
    # The -1 indicates a "wildcard" dimension, in this case the number of
    # images to use in the training.
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # 32 5x5 kernels, maintain resolution, relu
    #!!!CHANGE TO tf.keras.layers.conv2d to suppress warnings????
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5], # original
        #kernel_size=[4,4],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # 2x2 pool on 28x28 gives makes pool1 size 14x14
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # 64 5x5 kernels acting on 14x14 images, maintain resolution, relu
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5], # original
        #kernel_size=[4,4],
        padding="same",
        activation=tf.nn.relu)
        
    # Pooling Layer #2
    # 2x2 pool on 14x14 gives pool2 size 7x7
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense (fully connected) Layer
    # Resize 7x7x64 per input pool2 to be a single vector per input
    # dense and then dropout here are still large
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer (final classes)
    # Final layer to get down to 10 classes per input
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # If we are simply running a forward pass for prediction (actual runtime
    # usage), then just return that.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
      
      
      
# Run with defaults if at highest level
if __name__ == "__main__":  
    # Clear checkpoint files to get a clean training run each time
    checkpointSaveDir = "/tmp/mnist_convnet_model" # checkpoints saved here
    if os.path.exists(checkpointSaveDir): # only rm if it exists
        shutil.rmtree(checkpointSaveDir)    
    
    # Load training and eval data
    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    print("Number of training images: " + str(train_data.shape[0]))
    print("Number of testing  images: " + str(eval_data.shape[0]))
    print("Image shape (nx,ny): (" + str(train_data.shape[1]) + 
                               "," + str(train_data.shape[2]) + ")")

    train_data = train_data/np.float32(255) # normalize images
    train_labels = train_labels.astype(np.int32)  # not required

    eval_data = eval_data/np.float32(255) # normalize images
    eval_labels = eval_labels.astype(np.int32)  # not required
    
    # Display some example data
    chain = np.squeeze(train_data[0,:,:])
    for i in range(15):
        chain = np.append(chain,np.squeeze(train_data[i+1,:,:]),axis=1)
    print("Example training data")
    plt.figure()
    plt.imshow(chain)
    labelstrs = "".join([str(truth) + ", " for truth in train_labels[:16]])
    print("Truth: ", labelstrs)
    
    
    # Create the classifier object (estimator)
    # Note that THIS IS WHERE CNN_MODEL_FN GETS REFERENCED!!!
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model") #,
    #    config=checkptConfig)
        
       
    # Set up logging for predictions
    #tensors_to_log = {"probabilities": "softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(
    #    tensors=tensors_to_log, every_n_iter=50)
        
    # Configure the training function
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, # x is the training data
        y=train_labels, # y is the testing data
        batch_size=100, # number of images per batch (within an epoch)
        num_epochs=None,
        shuffle=True)

    # Train just one step and display the probabilties for sanity check
    #mnist_classifier.train(input_fn=train_input_fn,steps=1,hooks=[logging_hook])
    
    # Train for another 1000 epochs (do the grunt work)
    mnist_classifier.train(input_fn=train_input_fn, steps=1000)
    
    # Configure the evaluation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    # Run the evaluator and print the results
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

   


