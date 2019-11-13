# -*- coding: utf-8 -*-
"""
FILE: nfold_siamese_hourglass_cnn.py
DESCRIPTION:
    Reads in data and runs N-Fold cross validation on 
    train_siamese_hourglass_cnn. Outputs N trained networks (one for each fold)
    and conglomerates the performance of the four networks to give a full,
    complete, and unbiased performance analysis of the network.

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: November 2019
"""
import tensorflow as tf
import numpy as np
import videoUtilities as vu
import nnUtilities as nnu
import sys
import train_siamese_hourglass_cnn as tshc
from importData import importRoboticsLabData
import os
import shutil
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

# Write folded data to disk

# Reads all image files in and splits out into N folds.  Each fold goes into
# its own folder for later analysis.
def importFoldSave(N, saveDir):
    # Import the augmented robotics lab data sequences
    print("Reading augmented image and mask sequences")
    x_all, y_all, id_all, id_all_plus = importRoboticsLabData()
    
    
    # Split into train and test sets randomly
    print("Splitting into training and test sets randomly")
    x_folds, y_folds, id_folds, id_folds_plus = \
        vu.nFolding(NFolds, x_all, y_all, id_all, id_all_plus)
        
    # Save each fold of images to file
    for fold in range(N):
        foldDir = os.path.join(saveDir,"fold_%d" % (fold))
        
        # Clear the save directory to avoid collisions
        if os.path.exists(foldDir): # only rm if it exists
            shutil.rmtree(foldDir, ignore_errors=True) 
        os.mkdir(foldDir)
        
        # Loop over images and save
        nImages = x_folds[fold].shape[0]
        for iImage in range(nImages): 
            if iImage % 100 == 99:
                print("Fold %d: writing image %d of %d" % (fold, iImage+1, nImages))
            cv2.imwrite(os.path.join(foldDir,"image_" + id_folds_plus[fold][iImage] + '.jpg'), 
                        np.squeeze(x_folds[fold][iImage,:,:])*255)
            mask = np.squeeze(y_folds[fold][iImage,:,:]) < 1
            cv2.imwrite(os.path.join(foldDir,"mask_"  + id_folds_plus[fold][iImage] + '.jpg'),
                        mask*255)
        # end loop over images
    # end loop over folds
        
    # Convert masks to appropriately-weighted +/- masks
    print("Converting boolean masks into weighted +/- masks")
    y_folds_pmMask = []
    for fold in range(N):
        y_folds_pmMask.append(nnu.booleanMaskToPlusMinus(y_folds[fold], falseVal=-0.01))
    
    return x_folds, y_folds_pmMask, id_folds, id_folds_plus


# Read the images written out by the importFoldSave function above
def readFoldedImages(N,saveDir):
    x_folds = []
    y_folds = []
    id_folds = []
    id_folds_plus = []
    for fold in range(N):        
        # Read all the images in this fold's directory
        inImageBase = os.path.join(saveDir,"fold_%d" % fold,"image_")
        inMaskBase  = os.path.join(saveDir,"fold_%d" % fold,"mask_")
        imageStack, maskStack, indexStackPlus = vu.pull_aug_sequence(inImageBase, inMaskBase, ext='.jpg', color=False)
    
        # Add to the folds set
        x_folds.append(imageStack)
        y_folds.append(maskStack)
        id_folds_plus.append(indexStackPlus)
        indexStack = []
        for id_plus in indexStackPlus:
            indexStack.append(id_plus[:-3])
        id_folds.append(indexStack)
    # end for folds
    
    # Convert masks to appropriately-weighted +/- masks
    print("Converting boolean masks into weighted +/- masks")
    y_folds_pmMask = []
    for fold in range(N):
        y_folds_pmMask.append(nnu.booleanMaskToPlusMinus(y_folds[fold], falseVal=-0.01))
    
    return x_folds, y_folds_pmMask, id_folds, id_folds_plus
    

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
    nEpochs = 1000 #60000
    batchSize = 512
    NFolds = 4
    redoFolds = True # true to refold data
    saveDir = "folds" # directory to save folded data into
    
    if redoFolds:
        # Read in data, fold, and save to file
        x_folds, y_folds_pmMask, id_folds, id_folds_plus = \
            importFoldSave(NFolds, saveDir)
    else:
        # Read the folded data from file (as written previously by importFoldSave)
        x_folds, y_folds_pmMask, id_folds, id_folds_plus = \
            readFoldedImages(NFolds, saveDir)
     
    exit()
            
    # Form a set of arrays with all the images for later use
    nBatch, width, height = x_folds[0].shape[:3]
    x_all = np.zeros([0,width,height])
    y_all_pmMask = np.zeros([0,width,height])
    id_all = []
    id_all_plus = []
    for fold in range(NFolds):
        x_all = np.append(x_all,x_folds[fold],axis=0)
        y_all_pmMask = np.append(y_all_pmMask,y_folds_pmMask[fold],axis=0)
        id_all.extend(id_folds[fold])
        id_all_plus.extend(id_folds_plus[fold])
        
    # We can now loop over each fold of the input data, assigning it as the
    # test fold while training with the other folds.
    for fold in range(NFolds):
        x_train, y_train_pmMask = np.zeros([0,width,height]), np.zeros([0,width,height])
        id_train, id_train_plus = [], []
        
        # Deal out the training and test sets appropriately for this fold by
        # looping over all folds and putting this fold into test and the rest
        # into train
        for o in range(NFolds):
            if o == fold:
                x_test = x_folds[fold]
                y_test_pmMask = y_folds_pmMask[fold]
                id_test = id_folds[fold]
                id_test_plus = id_folds_plus[fold]
            else:
                x_train = np.append(x_train, x_folds[fold],axis=0)
                y_train_pmMask = np.append(y_train_pmMask, y_folds_pmMask[fold],axis=0)
                id_train.extend(id_folds[fold])
                id_train_plus.extend(id_folds_plus[fold])
        # for dealing out folds into test and train sets
        
        # Find all the siamese matches for each of the images
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
    
        # Run the statistics on this fold
        # We may want to parallelize the loop we are in here
        # We may also want to save the indices associated with each fold for
        # replicability.
        #!!! LEFT OFF HERE !!!
