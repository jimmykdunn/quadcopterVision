# -*- coding: utf-8 -*-
"""
FILE: use_hourglass_cnn_cv2
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
from datetime import datetime
import os
import videoUtilities as vu

"""
use_hourglass_cnn()
DESCRIPTION:
    Runs a forward pass of the tensorflow graph in modelPath on input images.
    Optionally runs multiple times to get more accurate timing statistics.
    This uses the cv2 library, not tensorflow.  This enables quicker and
    easier implementation on the SWAP-limited quadcopter's single board
    computers (odroid xu4 as of this writing).
    
INPUTS: 
    modelPath: protobuf (".pb") file to run a forward pass with
    inputImages: images to run forward pass on [numImages,width,height]

OPTIONAL INPUTS:
    numTimingTrials: number of times to repeat the same forward pass to get
        better timing statistics. Default 1.
    
RETURNS: 
    heatmaps: heatmaps that are the result of the forward pass in inputImages

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""
def use_hourglass_cnn(modelPath, inputImages, numTimingTrials = 1):
    # Execute a single forward pass on a set of images
    
    # Check this for meta to pb conversion and simply writing pb in the first place
    #https://stackoverflow.com/questions/48701666/save-tensorflow-checkpoint-to-pb-protobuf-file     
    print("Loading saved neural network from " + modelPath+'.pb')
    tensorflowNet = cv2.dnn.readNet(modelPath+'.pb')
    print("Neural network sucessfully loaded")
            
    # Run the forward pass for the input datapoints
    # In a real-time implementation, this will be called inside of
    # the data capture loop with single image datapoints. All of the
    # above preparation should be done ahead of time.
    start_time = datetime.now()
    for i in range(numTimingTrials): # run many times to get timing information
        #print("Setting input image with shape:")
        #print(inputImages.shape)
        tensorflowNet.setInput(np.squeeze(inputImages))
        #tensorflowNet.setInput(inputImages)
        #print("Input image set")
 
        # Runs a forward pass to compute the net output
        #print("Running forward pass")
        heatmaps = tensorflowNet.forward()
        #print("Forward pass complete")
 
    # Display timing trial information
    end_time = datetime.now()
    if numTimingTrials != 1:
        print("%d trials in %g seconds" % (numTimingTrials,(end_time-start_time).total_seconds()))
        print("Forward pass speed: %g Hz" % (numTimingTrials/(end_time-start_time).total_seconds()))
            
    return heatmaps


"""
quadcopterTest()
DESCRIPTION:
    Debugging function that runs a trained hourglass CNN on a basic quadcopter
    image.
    
INPUTS: 
    modelPath: protobuf (".pb") file to run a forward pass with
    
OUTPUTS: 
    Displays the resulting heatmap and saves to file

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
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
    #plt.imshow(joined)
    
    
"""
quadcopterBatchTest()
DESCRIPTION:
    Debugging function that runs a trained hourglass CNN on a series of basic 
    quadcopter images.
    
INPUTS: 
    modelPath: protobuf (".pb") file to run a forward pass with
    
OPTIONAL INPUTS:
    directory: directory containing the images to run the test on 
        (default goldenImages)
    ext: file extension of the images in directory (default .jpg)
    
OUTPUTS: 
    Saves the resulting heatmaps to file

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
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
   
    # Write the resulting filmstrip of heatmaps and overlays to file
    if not os.path.exists('heatmaps'):
        os.mkdir('heatmaps')
    cv2.imwrite(os.path.join('heatmaps','goldenFilmstrip.png'), filmstrip)
    print('Wrote ' + os.path.join('heatmaps','goldenFilmstrip.png'))
    #plt.imshow(filmstrip)
    #plt.show()
            
# Run with defaults if at highest level
if __name__ == "__main__":
    quadcopterBatchTest(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_64x48_20kepochs'))
