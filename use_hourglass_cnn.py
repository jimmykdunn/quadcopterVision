# -*- coding: utf-8 -*-
"""
FUNCTION: use_hourglass_cnn
DESCRIPTION:
    Uses a trained hourglass CNN on arbitrary input image(s)
    
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
import os

def use_hourglass_cnn(modelPath, inputImages, numTimingTrials = 1):
    # Execute a single forward pass on a set of images to demonstrate how
    # the trained classifier would be used in real-time software.
    
    # Initialize an empty graph
    graph = tf.Graph()
    
    # Just need an irrelevant placeholder for the b_masks "truth" variable
    placeholder_b_masks= np.zeros(np.shape(inputImages))

    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(modelPath)) # the ENTIRE session is now in saver
            saver.restore(sess,modelPath)
            
            # Get the entire dictionary of names (for debugging)
            #names = [i.name for i in sess.graph.get_operations()]
            
            # Get the output class probabilities function
            outputFn = graph.get_operation_by_name("heatmaps/b_heatmaps").outputs[0]
            b_images = graph.get_tensor_by_name("b_images:0")
            b_masks = graph.get_tensor_by_name("b_masks:0")
            
            # Run the forward pass for the input datapoints
            # In a real-time implementation, this will be called inside of
            # the data capture loop with single image datapoints. All of the
            # above preparation should be done ahead of time.
            start_sec = time.clock()
            for i in range(numTimingTrials): # run many times to get timing information
                heatmaps = sess.run(outputFn,feed_dict = {b_images: inputImages, b_masks: placeholder_b_masks})
            
                
            # Display timing trial information
            end_sec = time.clock()
            if numTimingTrials != 1:
                print("%d trials in %g seconds" % (numTimingTrials,end_sec-start_sec))
                print("Forward pass speed: %g Hz" % (numTimingTrials/(end_sec-start_sec)))
            
    return heatmaps

# Example of a two being drawn
def twoTest(modelPath):
    dataPath = os.path.join('.','mnist','jimmyDraws_2Blk.png')
    datapoint = np.mean(cv2.imread(dataPath),axis=2)/float(255)
    print("Running CNN at " + modelPath + " on " + dataPath)
    heatmap = use_hourglass_cnn(modelPath, 
                         np.reshape(datapoint,[1,datapoint.shape[0],datapoint.shape[1]]),
                         numTimingTrials=1000)
    
    # Join heatmap and actual image to a single array for output
    heatmapOutArray = np.squeeze(heatmap[0,:])*255.0
    testOutArray = np.squeeze(datapoint)*255.0
    joined = np.concatenate([testOutArray, heatmapOutArray],axis=0)
    cv2.imwrite(os.path.join('heatmaps','twoTest.png'), joined)
    print('Wrote ' + os.path.join('heatmaps','twoTest.png'))
    plt.imshow(joined)
    
# Example of a six being drawn
def sixTest(modelPath):
    dataPath = os.path.join('.','mnist','jimmyDraws_6Blk.png')
    datapoint = np.mean(cv2.imread(dataPath),axis=2)/float(255)
    print("Running CNN at " + modelPath + " on " + dataPath)
    heatmap = use_hourglass_cnn(modelPath, 
                         np.reshape(datapoint,[1,datapoint.shape[0],datapoint.shape[1]]),
                         numTimingTrials=1000)
    
    # Join heatmap and actual image to a single array for output
    heatmapOutArray = np.squeeze(heatmap[0,:])*255.0
    testOutArray = np.squeeze(datapoint)*255.0
    joined = np.concatenate([testOutArray, heatmapOutArray],axis=0)
    cv2.imwrite(os.path.join('heatmaps','sixTest.png'), joined)
    print('Wrote ' + os.path.join('heatmaps','sixTest.png'))
    plt.imshow(joined)
    
# Example quadcopter frame
def quadcopterTest(modelPath):
    dataPath = os.path.join('.','PHO_babyQuadcopterTest.jpg')
    datapoint = cv2.imread(dataPath)
    datapoint = np.mean(datapoint,axis=2)/float(255)
    print("Running CNN at " + modelPath + " on " + dataPath)
    heatmap = use_hourglass_cnn(modelPath, 
                         np.reshape(datapoint,[1,datapoint.shape[0],datapoint.shape[1]]),
                         numTimingTrials=100)
    
    # Join heatmap and actual image to a single array for output
    heatmapOutArray = np.squeeze(heatmap[0,:])*255.0
    testOutArray = np.squeeze(datapoint)*255.0
    joined = np.concatenate([testOutArray, heatmapOutArray],axis=0)
    cv2.imwrite(os.path.join('heatmaps','quadcopterTest.png'), joined)
    print('Wrote ' + os.path.join('heatmaps','quadcopterTest.png'))
    plt.imshow(joined)
            
# Run with defaults if at highest level
if __name__ == "__main__":
    
    #twoTest(os.path.join('mnist_hourglass_nn_save','model_at100.ckpt'))
    #print("\n\n")
    #sixTest(os.path.join('mnist_hourglass_nn_save','model_at100.ckpt'))
    print("\n\n")
    quadcopterTest(os.path.join('homebrew_hourglass_nn_save','model_at1000.ckpt'))
