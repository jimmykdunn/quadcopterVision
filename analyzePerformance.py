# -*- coding: utf-8 -*-
"""
FILE: analyzePerformance.py
DESCRIPTION:
    Functions for performance analysis on quadcopter drone heatmaps

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: November 2019
"""

from cv2.dnn import readNetFromTensorflow
import numpy as np
import os
from importData import importRoboticsLabData
import matplotlib.pyplot as plt
from nfold_siamese_hourglass_cnn import readFoldedImages


"""
calculateConfusionMatrix()
    Calculates the confusion matrix for the input heatmaps and associated truth
    masks via applying the input threshold to classify heatmap pixels as target
    or non-target.
INPUTS:
    heatmaps: array of heatmaps [nBatch,width,height]
    masks: array of truth masks [nBatch,width,height]
    threshold: threshold to apply. Heatmap pixels > threshold deemed target
RETURNS:
    2x2 confusion matrix [nbTruePos, nbFalsePos; nbFalseNeg, nbTrueNeg]
"""
# Confusion matrix calculation
def calculateConfusionMatrix(heatmaps, masks, threshold):

    # Apply the threshold to the heatmaps to generate a binary true (target) vs
    # false (non-target) for each pixel.
    heatmapClass = heatmaps > threshold
    maskClass = masks > 0
    
    # Compare classification of each pixel to the truth mask
    truePosPixels = np.logical_and(heatmapClass,maskClass);
    falsePosPixels = np.logical_and(heatmapClass,np.logical_not(maskClass));
    falseNegPixels = np.logical_and(np.logical_not(heatmapClass),maskClass);
    trueNegPixels = np.logical_and(np.logical_not(heatmapClass),np.logical_not(maskClass));
    
    # Calculate totals
    tp = np.sum(truePosPixels)
    fn = np.sum(falseNegPixels)
    fp = np.sum(falsePosPixels)
    tn = np.sum(trueNegPixels)
    
    return [tp,fn,fp,tn]
# end calculateConfusionMatrix


# Convert numerical confusion matrix to fractional
def confusionMatrixNumToPct(confusionMatrix):

    tpRate = confusionMatrix[0]/(confusionMatrix[0]+confusionMatrix[1]) # target called target/total targets
    fnRate = confusionMatrix[1]/(confusionMatrix[0]+confusionMatrix[1]) # target called noise/total targets
    fpRate = confusionMatrix[2]/(confusionMatrix[2]+confusionMatrix[3]) # noise called target/total noise
    tnRate = confusionMatrix[3]/(confusionMatrix[2]+confusionMatrix[3]) # noise called noise/total noise
    
    return [tpRate, fnRate, fpRate, tnRate]

# end confusionMatrixNumToPct
    
def runBasicPerformanceAnalysis(modelPath):
    # Import the augmented robotics lab data sequences
    print("Reading augmented image and mask sequences")
    x_all, y_all, id_all, id_all_plus = importRoboticsLabData(quickTest=True)
    #x_all, y_all, id_all, id_all_plus = importRoboticsLabData()
    
    # Import the trained neural network
    print("Loading saved neural network from " + modelPath+'.pb')
    tensorflowNet = readNetFromTensorflow(modelPath+'.pb')
    print("Neural network sucessfully loaded")
    
    # Test on a set of frames
    #tensorflowNet.setInput(x_all[:100,:,:])
    #heatmapsTest = tensorflowNet.forward() 
    
    # Execute a forward pass on all of the input frames
    heatmaps = np.zeros_like(x_all) # preallocate for speed
    for i, frame in enumerate(x_all):
        if i % 100 == 0:
            print("Running CNN on frame %d of %d" % (i, x_all.shape[0]))
        tensorflowNet.setInput(frame)
        heatmap = tensorflowNet.forward()
        heatmap = np.squeeze(heatmap)*255.0 # scale appropriately
        
        heatmaps[i,:,:] = heatmap # put into the big stack
        
    # Make the ROC Curves and confusion matrices on the heatmaps
    rocCurveAndConfusionMatrices(heatmaps, y_all, modelPath)
# end runBasicPerformanceAnalysis
    
    
'''
runNFoldPerformanceAnalysis()
    Run N-fold cross validation analysis with already-trained networks on the
    associated data.
INPUTS:
    N: number of folds. Nominally 4
    saveDir: directory where folded data are saved to. Nominally "folds".
    modelPath: location of trained networks. Will have "fold#" appended to it,
        where "#" is the number of the fold. Nominally "savedNetworks/[somename]"
OUTPUTS: 
    Prints the confusion matrices for each of a set of thresholds on the heatmaps.
    Plots and saves a ROC curve for the associated confusion matrices.
'''
def runNFoldPerformanceAnalysis(N, saveDir, modelPath, modelName="modelFinal_full"):
    # Read each fold of imagery
    x_folds, y_folds_pmMask, id_folds, id_folds_plus = readFoldedImages(N,saveDir)
    y_folds = [yfold > 0.0 for yfold in y_folds_pmMask]
    
    # Preallocate arrays for output and truth
    #nFrames_total = 0
    #for datafold in x_folds:
    #    nFrames_total += datafold.shape[0]
    #    
    #heatmaps_all = np.zeros((nFrames_total,x_folds[0].shape[1],x_folds[0].shape[2]))
    #y_pmMask_all = np.zeros((nFrames_total,x_folds[0].shape[1],x_folds[0].shape[2]))
    heatmaps_all = np.zeros((0,x_folds[0].shape[1],x_folds[0].shape[2]))
    y_all = np.zeros((0,x_folds[0].shape[1],x_folds[0].shape[2])).astype(np.bool)
    
    # Execute forward passes of each neural network on its respective fold of
    # data (which was trained with the other folds of data)
    for fold in range(N):
        foldModelPath = os.path.join(modelPath+"fold%d" % fold, modelName)
        #foldModelPath = os.path.join(modelPath+"fold%d" % fold,"model_at25000_full")
        
        # Import the trained neural network
        print("Loading saved neural network from " + foldModelPath+'.pb')
        tensorflowNet = readNetFromTensorflow(foldModelPath+'.pb')
        print("Neural network sucessfully loaded")
        
        # Execute a forward pass on all of the input frames for this fold
        heatmaps = np.zeros_like(x_folds[fold]) # preallocate for speed
        for i, frame in enumerate(x_folds[fold]):
            if i % 1000 == 0:
                print("Running CNN for fold %d on frame %d of %d" % (fold, i, x_folds[fold].shape[0]))
            tensorflowNet.setInput(frame)
            heatmap = tensorflowNet.forward()
            heatmap = np.squeeze(heatmap)*255.0 # scale appropriately
            
            heatmaps[i,:,:] = heatmap # put into the big stack
        
        # Append the full dataset arrays with this fold's arrays
        heatmaps_all = np.append(heatmaps_all,heatmaps,axis=0)
        y_all = np.append(y_all,y_folds[fold],axis=0)
        
        
    # We now have a complete array of heatmaps generated with nFold cross
    # validation methods and their associated truth masks. We can make 
    # confusion matrices and ROC curves with it.
    rocCurveAndConfusionMatrices(heatmaps_all, y_all, modelPath)

# end runNFoldPerformanceAnalysis
    
    
def rocCurveAndConfusionMatrices(heatmaps, y_all, modelPath):
    # Loop over a series of thresholds and calculate the resulting confusion
    # matrices and ROC curve points
    #thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresholds = np.arange(0,610,10) #[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300, 400, 500, 600]
    tpByThreshold = np.zeros_like(thresholds)
    fnByThreshold = np.zeros_like(thresholds)
    fpByThreshold = np.zeros_like(thresholds)
    tnByThreshold = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
    
        print("=============================")
        print("THRESHOLD: %g" % threshold)
        
        # Calculate the resulting confusion matrix
        confMatrix = calculateConfusionMatrix(heatmaps, y_all, threshold)
        
        # Print class statistics
        print('')
        print("Total number of target pixels: %11d" % (confMatrix[0]+confMatrix[1]))
        print("Total number of noise  pixels: %11d" % (confMatrix[2]+confMatrix[3]))
        print("#Noise Pixels/#TargetPixels = %g" % ((confMatrix[2]+confMatrix[3])/(confMatrix[0]+confMatrix[1])))
        
        # Print raw confusion matrix
        print('')
        print("Confusion matrix (pixel counts)")
        print("             call target,  call noise")
        print("true target  %11d, %11d" % (confMatrix[0], confMatrix[1]))
        print("true noise   %11d, %11d" % (confMatrix[2], confMatrix[3]))
        
        # Convert the confusion matrix to percentages and print
        print('')
        confMatrixPct = confusionMatrixNumToPct(confMatrix)
        print("Confusion matrix (percentages)")
        print("             call target,  call noise")
        print("true target        %5.2g,       %5.2g" % (confMatrixPct[0], confMatrixPct[1]))
        print("true noise         %5.2g,       %5.2g" % (confMatrixPct[2], confMatrixPct[3]))
        
        # Calculate right and wrong calls
        tpByThreshold[i] = confMatrix[0]
        fnByThreshold[i] = confMatrix[1]
        fpByThreshold[i] = confMatrix[2]
        tnByThreshold[i] = confMatrix[3]
    # end loop over thresholds
    
    # Make the ROC curve
    drawROCCurve(tpByThreshold, fnByThreshold, fpByThreshold, tnByThreshold)
    plt.savefig(modelPath + "_rocCurve.png")
    #plt.show()
# end rocCurveAndConfusionMatrices
    
    
    
    
def drawROCCurve(tpByThreshold, fnByThreshold, fpByThreshold, tnByThreshold):
    truePosRate  = tpByThreshold/(tpByThreshold+fnByThreshold)
    falseNegRate = fnByThreshold/(tpByThreshold+fnByThreshold)
    falsePosRate = fpByThreshold/(fpByThreshold+tnByThreshold)
    trueNegRate  = tnByThreshold/(fpByThreshold+tnByThreshold)
    
    plt.plot(falsePosRate,truePosRate,label='target pixel ROC')
    #plt.plot(falseNegRate,trueNegRate,label='background pixel ROC')
    #plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    
# Run if called directly
if __name__ == "__main__":
    #runBasicPerformanceAnalysis(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_mirror_sW00p50_1M00p00_2M00p00'))
    #runBasicPerformanceAnalysis(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_mirror_sW00p00_1M00p00_2M00p00'))
    
    # Quick test version (minimal data to read for debugging) (early network version)
    #runNFoldPerformanceAnalysis(4, 'testFolds', os.path.join('savedNetworks','noiseFix4Folds60k_sW00p00_'), modelName = "model_at25000_full")
    #runNFoldPerformanceAnalysis(4, 'testFolds', os.path.join('savedNetworks','noiseFix4Folds60k_sW00p50_'), modelName = "model_at25000_full")
    
    # Full version (reads all data) (uses final networks)
    #runNFoldPerformanceAnalysis(4, 'folds', os.path.join('savedNetworks','biasAdd4Folds60k_sW00p00_'), modelName = "modelFinal_full")
    runNFoldPerformanceAnalysis(4, 'folds', os.path.join('savedNetworks','biasAdd4Folds60k_sW00p50_'), modelName = "modelFinal_full")
