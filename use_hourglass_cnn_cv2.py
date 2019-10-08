# -*- coding: utf-8 -*-
"""
FUNCTION: use_hourglass_cnn_cv2
DESCRIPTION:
    Uses a trained hourglass CNN on arbitrary input image(s) using cv2 library
    
    Based on: https://jeanvitor.com/tensorflow-object-detecion-opencv/  

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
import videoUtilities as vu

def use_hourglass_cnn(modelPath, inputImages, numTimingTrials = 1):
    # Execute a single forward pass on a set of images to demonstrate how
    # the trained classifier would be used in real-time software.
    
    # Check this for meta to pb conversion and simply writing pb in the first place
    #https://stackoverflow.com/questions/48701666/save-tensorflow-checkpoint-to-pb-protobuf-file     
    tensorflowNet = cv2.dnn.readNet(modelPath+'.pb')
            
    # Run the forward pass for the input datapoints
    # In a real-time implementation, this will be called inside of
    # the data capture loop with single image datapoints. All of the
    # above preparation should be done ahead of time.
    start_sec = time.clock()
    for i in range(numTimingTrials): # run many times to get timing information
        
        tensorflowNet.setInput(np.squeeze(inputImages[0,:,:]))
 
        # Runs a forward pass to compute the net output
        heatmaps = tensorflowNet.forward()
 
    # Display timing trial information
    end_sec = time.clock()
    if numTimingTrials != 1:
        print("%d trials in %g seconds" % (numTimingTrials,end_sec-start_sec))
        print("Forward pass speed: %g Hz" % (numTimingTrials/(end_sec-start_sec)))
            
    return heatmaps

# Example quadcopter frame
def quadcopterTest(modelPath):
    dataPath = os.path.join('.','PHO_quadcopterTest_64x64.jpg')
    datapoint = cv2.imread(dataPath)
    datapoint = np.mean(datapoint,axis=2)/float(255)
    print("Running CNN at " + modelPath + " on " + dataPath)
    heatmap = use_hourglass_cnn(modelPath, 
                         np.reshape(datapoint,[1,datapoint.shape[0],datapoint.shape[1]]),
                         numTimingTrials=100)
    
    # Overlay on outline of the heatmap in green onto the image
    greenedImage = vu.overlay_heatmap(heatmap,datapoint)
        
    # Join heatmap and actual image to a single array for output
    heatmapOutArray = np.squeeze(heatmap[0,:])*255.0
    heatmapOutArray = np.minimum(heatmapOutArray,np.ones(heatmapOutArray.shape)*255)
    heatmapOutArray = np.maximum(heatmapOutArray,np.zeros(heatmapOutArray.shape))
    heatmapOutArray = heatmapOutArray.astype(np.uint8)
    heatmapOutArray = np.repeat(heatmapOutArray[:,:,np.newaxis],3,axis=2)
    joined = np.concatenate([greenedImage, heatmapOutArray],axis=0)
    cv2.imwrite(os.path.join('heatmaps','quadcopterTest.png'), joined)
    print('Wrote ' + os.path.join('heatmaps','quadcopterTest.png'))
    plt.imshow(joined)
    
    
# Example quadcopter frames
def quadcopterBatchTest(modelPath,directory='goldenImages',ext='.jpg'):
    iImage = 0
    filmstrip = []
    # Loop over each file in the path
    for filename in os.listdir(directory):
        # Pull only if the image extension is in the filename
        tokens = filename.split('_')
        prefix = tokens[0]
        parent = tokens[1]
        augment = tokens[2].split('.')[0]
        fileext = tokens[2].split('.')[1]
        if ext in filename and 'mage' in prefix:
            datapoint = cv2.imread(os.path.join(directory,filename))
            # Find corresponding mask if it is there
            maskName = os.path.join(directory,'augMask_'+parent+'_'+augment+'.'+fileext)
            if os.path.isfile(maskName):
                maskpoint = cv2.imread(maskName)
                
                datapoint = np.mean(datapoint,axis=2)/float(255)
                maskpoint = np.mean(maskpoint,axis=2) == 0
                print("Running CNN at " + modelPath + " on " + os.path.join(directory,filename))
                heatmap = use_hourglass_cnn(modelPath, 
                                     np.reshape(datapoint,[1,datapoint.shape[0],datapoint.shape[1]]),
                                     numTimingTrials=100)
                
                # Overlay on outline of the heatmap in green onto the image
                #greenedImage = vu.overlay_heatmap(heatmap,datapoint)
                greenedImage = vu.overlay_heatmap_and_mask(heatmap,maskpoint,datapoint)
                   
                # Overlay heatmap and mask center of mask onto the image
                heatmapCOM = vu.find_centerOfMass(heatmap)
                greenedImage = vu.overlay_point(greenedImage,heatmapCOM,color='g')
                maskCOM = vu.find_centerOfMass(maskpoint)
                greenedImage = vu.overlay_point(greenedImage,maskCOM,   color='r')
                
                # Join heatmap and actual image to a single array for output
                heatmapOutArray = np.squeeze(heatmap[0,:])*255.0
                heatmapOutArray = np.minimum(heatmapOutArray,np.ones(heatmapOutArray.shape)*255)
                heatmapOutArray = np.maximum(heatmapOutArray,np.zeros(heatmapOutArray.shape))
                heatmapOutArray = heatmapOutArray.astype(np.uint8)
                heatmapOutArray = np.repeat(heatmapOutArray[:,:,np.newaxis],3,axis=2)
                pair = np.concatenate([greenedImage, heatmapOutArray],axis=0)
                
                # Create filmstrip if this is the first image in the folder, 
                # otherwise tack it on
                if iImage == 0:
                    filmstrip = pair
                else:
                    filmstrip = np.concatenate([filmstrip,pair],axis=1)
                    
                # Increment counter
                iImage += 1
            # if corresponding mask exists
        # if this file is an image
    # for all files in directory
    
    
    cv2.imwrite(os.path.join('heatmaps','goldenFilmstrip.png'), filmstrip)
    print('Wrote ' + os.path.join('heatmaps','goldenFilmstrip.png'))
    #plt.imshow(filmstrip)
    #plt.show()
            
# Run with defaults if at highest level
if __name__ == "__main__":
    
    #print("\n\n")
    #quadcopterTest(os.path.join('homebrew_hourglass_nn_save','model_at750.ckpt'))
    quadcopterBatchTest(os.path.join('homebrew_hourglass_nn_save','modelFinal'))
