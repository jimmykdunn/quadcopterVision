# -*- coding: utf-8 -*-
"""
FUNCTION: use_cnn
DESCRIPTION:
    Uses a trained CNN on a single data example 
    
    Based on: https://datascience.stackexchange.com/questions/16922/using-tensorflow-model-for-prediction
    
INPUTS: 

    
OUTPUTS: 
    

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
import cv2
import time

def use_cnn(modelPath, datapoints, numTimingTrials = 1):
    # Execute a single forward pass on a single image to demonstrate how
    # the trained classifier would be used in real-time software.
    
    # Initialize a braph
    graph = tf.Graph()
    
    # Just need an irrelevant placeholder for the y_ "truth" variable
    placeholderY = np.zeros([len(datapoints),10]).astype(np.int32)

    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(modelPath)) # the ENTIRE session is now in saver
            saver.restore(sess,modelPath)
            
            # Get the entire dictionary of names (for debugging)
            #names = [i.name for i in sess.graph.get_operations()]
            
            # Get the output class probabilities function
            outputFn = graph.get_operation_by_name("outputClassProbs/classProbs").outputs[0]
            x = graph.get_tensor_by_name("x:0")
            y_ = graph.get_tensor_by_name("y_:0")
            
            # Run the forward pass for the input datapoints
            # In a real-time implementation, this will be called inside of
            # the data capture loop with single image datapoints. All of the
            # above preparation should be done ahead of time.
            start_sec = time.clock()
            for i in range(numTimingTrials): # run many times to get timing information
                outputClassProbs = sess.run(outputFn,feed_dict = {x: datapoints, y_: placeholderY})
            
                # Calculate and print the results
                #print(outputClassProbs)
                predictions = np.argmax(outputClassProbs,1)
                #print("The input datapoints look like: ")
                #print(predictions)
                
            # Display timing trial information
            end_sec = time.clock()
            if numTimingTrials != 1:
                print("%d trials in %g seconds" % (numTimingTrials,end_sec-start_sec))
                print("Forward pass speed: %g Hz" % (numTimingTrials/(end_sec-start_sec)))
            
    return predictions

# Example of a two being drawn
def twoTest(modelPath):
    dataPath = './jimmyDraws_2Blk.png'
    datapoint = np.mean(cv2.imread(dataPath),axis=2)/float(255)
    print("Running CNN at " + modelPath + " on " + dataPath)
    plt.imshow(datapoint)
    prediction = use_cnn(modelPath, 
                         np.reshape(datapoint,[1,datapoint.shape[0],datapoint.shape[1]]),
                         numTimingTrials=1000)
    print(dataPath + " appears to be a " + str(prediction))
    
# Example of a siz being drawn
def sixTest(modelPath):
    dataPath = './jimmyDraws_6Blk.png'
    datapoint = np.mean(cv2.imread(dataPath),axis=2)/float(255)
    print("Running CNN at " + modelPath + " on " + dataPath)
    plt.imshow(datapoint)
    prediction = use_cnn(modelPath, 
                         np.reshape(datapoint,[1,datapoint.shape[0],datapoint.shape[1]]),
                         numTimingTrials=1000)
    print(dataPath + " appears to be a " + str(prediction))
            
# Run with defaults if at highest level
if __name__ == "__main__":
    twoTest("./mnist_cnn_save/model_at1000.ckpt")
    print("\n\n")
    sixTest("./mnist_cnn_save/model_at1000.ckpt")
    
