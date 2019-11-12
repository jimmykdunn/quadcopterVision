# -*- coding: utf-8 -*-
"""
FILE: run_siamese_hourglass_cnn.py
DESCRIPTION:
    Reads in data and runs a single instance of train_siamese_hourglass_cnn.py.
    Does NOT use full N-Fold cross validation.

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""
import tensorflow as tf
import numpy as np
import os
import cv2
import copy
import videoUtilities as vu
import nnUtilities as nnu
import sys
import train_siamese_hourglass_cnn as tshc
from importData import importRoboticsLabData

tf.logging.set_verbosity(tf.logging.INFO)


"""
Build and train the hourglass CNN from the main level when this file is called.
"""

if __name__ == "__main__":  

    print("Running:")
    for arg in sys.argv:
        print(arg)
        
    # Parse command line inputs
    if len(sys.argv) < 2:
        siameseWeight = 0.0 # default
    else:
        siameseWeight = float(sys.argv[1])
    
    if len(sys.argv) < 3:
        firstMomentWeight = 0.0 # default
    else:
        firstMomentWeight = float(sys.argv[2])
        
    if len(sys.argv) < 4:
        secondMomentWeight = 0.0 # default
    else:
        secondMomentWeight = float(sys.argv[3])
        
    if len(sys.argv) < 5:
        saveName = "savedNetwork" # default
    else:
        saveName = sys.argv[4]
        
    print("siameseWeight = %g" % siameseWeight)
    print("firstMomentWeight = %g" % firstMomentWeight)
    print("secondMomentWeight = %g" % secondMomentWeight)
    print("saveName = " + saveName)
    
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    # Set additional default parameters
    checkpointSaveDir = "./savedNetworks/" + saveName
    peekEveryNEpochs=1000
    saveEveryNEpochs=1000
    nEpochs = 60000
    batchSize = 512
    
    # Import the augmented robotics lab data sequences
    print("Reading augmented image and mask sequences")
    x_all, y_all, id_all, id_all_plus = importRoboticsLabData()
    
    # Convert masks to appropriately-weighted +/- masks
    print("Converting boolean masks into weighted +/- masks")
    y_all_pmMask = nnu.booleanMaskToPlusMinus(y_all, falseVal=-0.01)
    
    # Split into train and test sets randomly
    print("Splitting into training and test sets randomly")
    x_train, y_train_pmMask, x_test, y_test_pmMask, id_train, id_test, id_train_plus, id_test_plus = \
        vu.train_test_split_noCheat(x_all, y_all_pmMask, id_all, id_all_plus, trainFraction=0.8)
    
        
    # Find all the siamese matches for each of the training and testing images
    # Allocate memory ahead of time for speed
    print("Allocating memory for both arms of the Siamese network")
    x_trainA, y_train_pmMaskA, x_testA, y_test_pmMaskA = \
        np.zeros_like(x_train), np.zeros_like(y_train_pmMask), np.zeros_like(x_test), np.zeros_like(y_test_pmMask)
    x_trainB, y_train_pmMaskB, x_testB, y_test_pmMaskB =  \
        np.zeros_like(x_train), np.zeros_like(y_train_pmMask), np.zeros_like(x_test), np.zeros_like(y_test_pmMask)
    
    
    # Find all existing training image matches
    print("Finding siamese matches for training images")
    numFound = 0
    for i in range(len(id_train_plus)):
        # Search thru all images - simaese match may not be in same set
        pairedImage, pairedMask, pairedIndexString = vu.find_siamese_match(
            id_train_plus[i], x_all, y_all_pmMask, id_all, id_all_plus, randomSign=True)
        if pairedIndexString != "****_****": # if there is a siamese match
            x_trainA[numFound,:,:] = x_train[i,:,:]
            y_train_pmMaskA[numFound,:,:] = y_train_pmMask[i,:,:]
            x_trainB[numFound,:,:] = pairedImage
            y_train_pmMaskB[numFound,:,:] = pairedMask
            numFound += 1
        else:
            print("Match not found for train image " + id_train[i])
    # end for training images
    # Cut off the excess
    x_trainA = x_trainA[:numFound,:,:]
    y_train_pmMaskA = y_train_pmMaskA[:numFound,:,:]
    x_trainB = x_trainB[:numFound,:,:]
    y_train_pmMaskB = y_train_pmMaskB[:numFound,:,:]
    
    
    # Find all existing testing image matches
    print("Finding siamese matches for test images")
    numFound = 0
    for i in range(len(id_test_plus)):
        # Search thru all images - simaese match may not be in same set
        pairedImage, pairedMask, pairedIndexString = vu.find_siamese_match(
            id_test_plus[i], x_all, y_all_pmMask, id_all, id_all_plus)
        if pairedIndexString != "****_****": # if there is a siamese match
            x_testA[numFound,:,:] = x_test[i,:,:]
            y_test_pmMaskA[numFound,:,:] = y_test_pmMask[i,:,:]
            x_testB[numFound,:,:] = pairedImage
            y_test_pmMaskB[numFound,:,:] = pairedMask
            numFound += 1
        else:
            print("Match not found for test image " + id_test[i])
    # end for test images
    # Cut off the excess
    x_testA = x_testA[:numFound,:,:]
    y_test_pmMaskA = y_test_pmMaskA[:numFound,:,:]
    x_testB = x_testB[:numFound,:,:]
    y_test_pmMaskB = y_test_pmMaskB[:numFound,:,:]
    
    
    # Run the complete training on the hourglass neural net
    print("Running siamese hourglass training")
    heatmaps = tshc.train_siamese_hourglass_cnn(
            x_trainA, y_train_pmMaskA, x_testA, y_test_pmMaskA, 
            x_trainB, y_train_pmMaskB, x_testB, y_test_pmMaskB, 
        checkpointSaveDir = checkpointSaveDir, peekEveryNEpochs = peekEveryNEpochs,
        saveEveryNEpochs=saveEveryNEpochs, nEpochs=nEpochs, batchSize=batchSize,
        siameseWeight=siameseWeight, firstMomentWeight=firstMomentWeight, secondMomentWeight=secondMomentWeight)
    
    '''
    # Write out the first few testset heatmaps to file along with the associated
    # test data inputs for visualization
    print("Generating some sample heatmap images for visualization")
    if not os.path.isdir('heatmaps'): # make the output dir if needed
        os.mkdir('heatmaps')
    numToWrite = np.min([16,heatmaps.shape[0]])
    filmstrip = []
    for iHeat in range(numToWrite):
        # Make the output images individually
        heatmapOutArray = np.squeeze(heatmaps[iHeat,:])
        testOutArray = np.squeeze(x_test[iHeat,:])
        maskOutArray = np.squeeze(y_test[iHeat,:])
        
        # Overlay contour lines for thresholded heatmap and mask
        testOutArray = vu.overlay_heatmap_and_mask(heatmapOutArray,maskOutArray,testOutArray)
        heatmapCOM = vu.find_centerOfMass(heatmapOutArray)
        testOutArray = vu.overlay_point(testOutArray,heatmapCOM,color='g')
        maskCOM = vu.find_centerOfMass(maskOutArray)
        testOutArray = vu.overlay_point(testOutArray,maskCOM,   color='r')
        
        # Join heatmap and actual image to a single array for output
        joinedStr = 'joined_%04d.png' % iHeat
        heatmapOutArrayCol = np.repeat(255*heatmapOutArray[:,:,np.newaxis],3,axis=2)
        joined = np.concatenate([testOutArray, heatmapOutArrayCol],axis=0)
        cv2.imwrite(os.path.join('heatmaps',joinedStr), joined)
        print('Wrote ' + os.path.join('heatmaps',joinedStr))
        
        # Make output strip of images and heatmaps
        if iHeat == 0:
            filmstrip = copy.deepcopy(joined)
        filmstrip = np.concatenate([filmstrip,joined], axis=1)
        
    # Write all numToWrite in a single image for easy analysis
    cv2.imwrite(os.path.join('heatmaps','filmstrip.png'), filmstrip) 
    print('Wrote ' + os.path.join('heatmaps','filmstrip.png')) 
    '''
        
