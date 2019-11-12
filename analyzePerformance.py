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
    
    
    # Loop over a series of thresholds and calculate the resulting confusion
    # matrices and ROC curve points
    #thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600]
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
    plt.show()
# end runBasicPerformanceAnalysis
    
    
def drawROCCurve(tpByThreshold, fnByThreshold, fpByThreshold, tnByThreshold):
    truePosRate  = tpByThreshold/(tpByThreshold+fnByThreshold)
    falseNegRate = fnByThreshold/(tpByThreshold+fnByThreshold)
    falsePosRate = fpByThreshold/(fpByThreshold+tnByThreshold)
    trueNegRate  = tnByThreshold/(fpByThreshold+tnByThreshold)
    
    plt.plot(falsePosRate,truePosRate,label='target pixel ROC')
    plt.plot(falseNegRate,trueNegRate,label='background pixel ROC')
    plt.legend()
    plt.xlabel("False classification rate")
    plt.ylabel("True classification rate")
    
# Run if called directly
if __name__ == "__main__":
    #runBasicPerformanceAnalysis(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_mirror_sW00p50_1M00p00_2M00p00'))
    runBasicPerformanceAnalysis(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_mirror_sW00p00_1M00p00_2M00p00'))
