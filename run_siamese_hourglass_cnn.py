# -*- coding: utf-8 -*-
"""
FILE: run_siamese_hourglass_cnn.py
DESCRIPTION:
    Trains and evaluates a convolutional neural net (CNN) to generate a heatmap
    of the probability of a quadcopter being at each pixel in the input image.
    Uses siamese structure to smooth the output heatmaps.
    Uses tensorflow.  

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
import shutil
import time
import cv2
import copy
import videoUtilities as vu
import matplotlib.pyplot as plt
import nnUtilities as nnu
import neuralNetStructures as nns
import sys

tf.logging.set_verbosity(tf.logging.INFO)


"""
Train the siamese hourglass NN. Makes the graph with hourglass_nn, then 
calculates the heatmaps for each of two input data feeds, calculates gain
as the sum of the difference between the mask and the heatmap and the difference
between the heatmaps of the two siamese networks. Ends with a test forward pass.
"-1" is a placeholder for the number of images in the batch.
INPUTS:
    trainImagesA: input images, [-1,nx,ny,1] 
    trainMasksA: truth masks associated with trainImages. +1 if target, -1 otherwise, [-1,nx,ny] 
    testImagesA: test images, [-1,nx,ny,1]
    testMasksA: truth masks associated with testImages. +1 if target, -1 otherwise, [-1,nx,ny] 
    trainImagesB: input images for 2nd arm of Siamese network, [-1,nx,ny,1]
    trainMasksB: truth masks for 2nd arm of Siamese network associated with 
                 trainImages. +1 if target, -1 otherwise, [-1,nx,ny] 
    testImagesB: test images for 2nd arm of Siamese network, [-1,nx,ny,1]
    testMasksB: truth masks for 2nd arm of Siamese network associated with 
                 testImages. +1 if target, -1 otherwise, [-1,nx,ny]
OPTIONAL INPUTS:
    nEpochs: number of training epcohs to run (default 100)
    batchSize: number of images to use per training epoch (default 100)
    checkpointSaveDir: directory to save trained neural net and graph to (default './hourglass_nn_save")
    saveEveryNEpochs: save checkpoint every this many epochs (and at the end) (default 500)
    peekEveryNEpochs: print the training gain with a forward pass every this many epochs (default 50)
    siameseWeight: weight to apply to siamese deltas relative to heatmap gain (default 1.0)
    firstMomentWeight: weight to apply to first moment loss relative to heatmap gain (default 1.0)
    secondMomentWeight: weight to apply to second moment loss relative to heatmap gain (default 1.0)
EXAMPLE:
    test_heatmaps = train_hourglass_nn(x)
RETURNS:
    test_heatmaps: map of pixel values for each image, higher for pixels more 
                   likely to be target. Same size as input testImages.
"""
def train_siamese_hourglass_nn(trainImagesA, trainMasksA, testImagesA, testMasksA, \
                               trainImagesB, trainMasksB, testImagesB, testMasksB, \
    nEpochs=100, batchSize=100, checkpointSaveDir='./hourglass_nn_save', \
    saveEveryNEpochs=500, peekEveryNEpochs=50, siameseWeight = 1.0,
    firstMomentWeight = 1.0, secondMomentWeight = 1.0):
    
    print("BEGIN SIAMESE HOURGLASS NN TRAINING")
    
    # Clear checkpoint files to get a clean training run each time
    if os.path.exists(checkpointSaveDir): # only rm if it exists
        shutil.rmtree(checkpointSaveDir, ignore_errors=True)   
    else:
        os.mkdir(checkpointSaveDir)
        
    # Image sizes
    nBatch,  nWidth,  nHeight  = trainImagesA.shape[:3]
    nBatchB, nWidthB, nHeightB = trainImagesB.shape[:3]
    
    # Check that the two arms of the Siamese network have identical shapes
    assert nBatch  == nBatchB
    assert nWidth  == nWidthB
    assert nHeight == nHeightB

    # Placeholders for the data and associated truth
    # "b_" prefix stands for "batch"
    with tf.name_scope('inputs'):
        b_images = tf.placeholder(tf.float32, [None, nWidth,nHeight], name="b_images")
        b_masks = tf.placeholder(tf.float32, [None, nWidth,nHeight], name="b_masks")
    
    # Build the graph for the deep hourglass net
    # It is best to literally thing of this as just building the graph, since
    # it is in reality just a "placeholder" or template for what we will
    # actually be running.
    with tf.name_scope('heatmaps'):
        # b_images here is the concatenation of both arms of the siamese
        # network, but we will only use the heatmaps from the first half
        # (the first arm) as the output
        b_heatmapsAll = nns.hourglass_nn(b_images)
        b_heatmapsAll = tf.reshape(b_heatmapsAll,[-1,nWidth,nHeight],'b_heatmaps')
        
    with tf.name_scope('siameseSplit'):
        # Split off into siamese halves
        heatmapShape = tf.shape(b_heatmapsAll)
        nBatch = tf.cast(tf.divide(heatmapShape[0],2),tf.int32)
        b_heatmapsA = b_heatmapsAll[:nBatch,:,:] # arm A heatmaps
        b_heatmapsB = b_heatmapsAll[nBatch:,:,:] # arm B heatmaps
        b_masksA = b_masks[:nBatch,:,:] # arm A masks
        #b_masksB = b_masks[nBatch:,:,:] # arm B masks
   
    # The heatmap loss calculation
    with tf.name_scope('heatmapGain'):
        # Gain here is really just an pixel-wise heatmap*truthmask product
        # + gain for every heatmap pixel that IS     part of the targetmask
        # - gain for every heatmap pixel that IS NOT part of the targetmask
        # To do this, targetmask must have +1's at target     locations
        # and         targetmask must have -1's at background locations
        # Make sure targetmask is formed in this way!!!
        b_gainmaps = tf.multiply(b_heatmapsA, b_masksA) # pixel-by-pixel gain
        b_gainmaps = tf.math.minimum(b_gainmaps, 1.0, name="b_gainmaps") # anything above 1 doesn't help
        
        # May be useful to have an intermediate reduction here of a single
        # gain value for each individual image...
        
        # Average of gain across every pixel of every image
        heatmapGain = tf.reduce_mean(tf.cast(b_gainmaps,tf.float32))
        heatmapLoss = tf.multiply(-1.0,heatmapGain)
        
        # Perfect segementation would result in this gain value
        booleanMask = tf.math.greater(b_masks,0)
        perfectGain = tf.reduce_mean(tf.cast(booleanMask,tf.float32))
    
    with tf.name_scope('siameseGain'):    
        # Calculate difference maps between each arm of the siamese network.
        # Square to make positive and for mathematical niceness.
        b_siameseDelta = tf.subtract(b_heatmapsA,b_heatmapsB)
        b_siameseDelta = tf.square(b_siameseDelta)
        siameseLoss = tf.reduce_mean(tf.cast(b_siameseDelta,tf.float32))
        
    with tf.name_scope('momentLoss'):
        # Calculate first moment loss
        firstMomentLoss, xCOM, yCOM, xMaskCOM, yMaskCOM = nnu.calculateFirstMomentLoss(b_heatmapsA,b_masksA)
        
        # Calculate second moment loss
        secondMomentLoss, xSTD, ySTD = nnu.calculateSecondMomentLoss(b_heatmapsA,b_masksA)
        
    with tf.name_scope('overallLoss'):
        # Overall loss is just the weighted sum of all the loss terms
        loss = tf.add(heatmapLoss, 
                      tf.multiply(siameseWeight, siameseLoss))
        #loss = tf.add(
        #       tf.add(heatmapLoss, 
        #              tf.multiply(siameseWeight, siameseLoss)),
        #       tf.add(tf.multiply(firstMomentWeight,  firstMomentLoss),
        #              tf.multiply(secondMomentWeight, secondMomentLoss)))
                      
        
        
    # Optimization calculation
    with tf.name_scope('adam_optimizer'):
        # Basic ADAM optimizer
        train_step = tf.train.AdamOptimizer().minimize(loss)
        
        
    # Save the graph of the neural network and loss function
    print('Saving graph to: %s' % checkpointSaveDir)
    train_writer = tf.summary.FileWriter(checkpointSaveDir)
    train_writer.add_graph(tf.get_default_graph())
    
    # Initialize class to save the CNN post-training
    # Save up to 100 along the way for comparisons
    saver = tf.train.Saver(max_to_keep=100)
    
    # Start the clock
    start_sec = time.clock()
    
    # Actually execute the training using the CNN template we just built
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Loop over every epoch
        peekSchedule = []
        trainGainHistory = []
        testGainHistory = []
        for epoch in range(nEpochs): 
            # Extract data for this batch
            batchA = nnu.extractBatch(batchSize, trainImagesA, trainMasksA, epoch)
            batchB = nnu.extractBatch(batchSize, trainImagesB, trainMasksB, epoch)
            
            # Run a single epoch with the extracted data batch
            batchImages = np.concatenate([batchA[0], batchB[0]], axis=0) # concatenate siamese arms
            batchMasks = np.concatenate([batchA[1], batchB[1]], axis=0) # concatenate siamese arms
            train_step.run(feed_dict={b_images: batchImages, b_masks: batchMasks}) 
            
            # Check our progress on the training data every peekEveryNEpochs epochs
            if epoch % peekEveryNEpochs == (peekEveryNEpochs-1):
                trainHeatmapLoss   = heatmapLoss.eval(     feed_dict={b_images: batchImages, b_masks: batchMasks})
                trainSiameseLoss   = siameseLoss.eval(     feed_dict={b_images: batchImages, b_masks: batchMasks})
                train1stMomentLoss = firstMomentLoss.eval( feed_dict={b_images: batchImages, b_masks: batchMasks})
                train2ndMomentLoss = secondMomentLoss.eval(feed_dict={b_images: batchImages, b_masks: batchMasks})
                trainTotalLoss     = loss.eval(            feed_dict={b_images: batchImages, b_masks: batchMasks})
                print('epoch %d of %d, training heatmap loss %g, training siamese loss %g, training 1st moment loss %g, training 2nd moment loss %g, training total loss %g' % (epoch+1, nEpochs, trainHeatmapLoss, trainSiameseLoss, train1stMomentLoss, train2ndMomentLoss, trainTotalLoss))
                testBatch = nnu.extractBatch(100, testImagesA, testMasksA, 0, randomDraw=True)
                testHeatmapLoss = heatmapLoss.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                testHeatmapGain = heatmapGain.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                perfectTestGain = perfectGain.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                print('epoch %d of %d, test heatmap gain (max 1.0) %g, testHeatmapLoss %g' % (epoch+1, nEpochs, testHeatmapGain/perfectTestGain, testHeatmapLoss))
                
                '''
                # DEBUGGUNG PEEKS
                heatmapsAll = b_heatmapsA.eval(feed_dict={b_images: batchImages, b_masks: batchMasks})
                xCOMAll = xCOM.eval(feed_dict={b_images: batchImages, b_masks: batchMasks})
                yCOMAll = yCOM.eval(feed_dict={b_images: batchImages, b_masks: batchMasks})
                xMaskCOMAll = xMaskCOM.eval(feed_dict={b_images: batchImages, b_masks: batchMasks})
                yMaskCOMAll = yMaskCOM.eval(feed_dict={b_images: batchImages, b_masks: batchMasks})
                xSTDAll = xSTD.eval(feed_dict={b_images: batchImages, b_masks: batchMasks})
                ySTDAll = ySTD.eval(feed_dict={b_images: batchImages, b_masks: batchMasks})
                print('heatmaps')
                print(heatmapsAll[0,:,:])
                print('xCOM')
                print(xCOMAll[:20])
                print('yCOM')
                print(yCOMAll[:20])
                print('xMaskCOM')
                print(xMaskCOMAll[:20])
                print('yMaskCOM')
                print(yMaskCOMAll[:20])
                print('xSTD')
                print(xSTDAll[:20])
                print('ySTD')
                print(ySTDAll[:20])
                # DEBUGGUNG PEEKS
                '''
                
                peekSchedule.append(epoch+1)
                trainGainHistory.append(-trainHeatmapLoss)
                testGainHistory.append(-testHeatmapLoss)
            
            # Print elapsed time every peekEveryNEpochs epochs
            if epoch % peekEveryNEpochs == (peekEveryNEpochs-1):
                curr_sec = time.clock()
                print('    Elapsed time for %d epochs: %g sec' 
                      % (epoch+1, curr_sec-start_sec))
    
            # Save the model weights (and everything else) every 
            # saveEveryNEpochs epochs, but always save at the end.
            if epoch % saveEveryNEpochs == (saveEveryNEpochs-1) or epoch == (nEpochs-1):
                save_path = saver.save(sess, checkpointSaveDir + "/model_at" + str(epoch+1) + ".ckpt")
                print("    Checkpoint saved to: %s" % save_path)
                
        # Save graph and fully trained model as a protobuf       
        nnu.save_graph_protobuf(sess,checkpointSaveDir)
        
        print("\n\n\n\n")
        print("============================")
        print("TRAINING RESULTS")
        print("============================")

        # Total elapsed time
        end_sec = time.clock()
        print('Total elapsed time for %d epochs: %g sec' 
              % (nEpochs, end_sec-start_sec))
    
        # Finish off by running the test set.  Extract the entire test set.
        test_batch = nnu.extractBatch(len(testImagesA), testImagesA, testMasksA, 0)
        test_gain = heatmapGain.eval(feed_dict={b_images: test_batch[0], b_masks: test_batch[1]})
        perfectTestGain = perfectGain.eval(feed_dict={b_images: test_batch[0], b_masks: test_batch[1]})
        test_heatmaps = b_heatmapsAll.eval(feed_dict={b_images: test_batch[0], b_masks: test_batch[1]})
        print('test gain %g' % (test_gain/perfectTestGain))
        
        # Print the location of the saved network
        print("Final trained network saved to: " + save_path)
        print("You can use use_cnn.py with this final network to classify new datapoints")
        
    # Generate a plot of the train vs test gain as a function of epoch
    plt.plot(peekSchedule,trainGainHistory,'bo', linestyle='-')
    plt.plot(peekSchedule,testGainHistory,'go', linestyle='-')
    plt.legend(['Train Gain','Test Gain'])
    plt.xlabel('Epoch')
    plt.ylabel('Gain')
    plt.savefig(os.path.join('heatmaps','trainTestHistory.png'))
    print("Wrote trainTestHistory to " + os.path.join('heatmaps','trainTestHistory.png'))
    #plt.show()
        
    return test_heatmaps
    

"""
Build and train the hourglass CNN from the main level when this file is called.
"""

if __name__ == "__main__":  

    print("Running:")
    for arg in sys.argv:
        print(arg)
    siameseWeight = float(sys.argv[1])
    firstMomentWeight = float(sys.argv[2])
    secondMomentWeight = float(sys.argv[3])
    saveName = sys.argv[4]
    print("siameseWeight = %g" % siameseWeight)
    print("firstMomentWeight = %g" % firstMomentWeight)
    print("secondMomentWeight = %g" % secondMomentWeight)
    print("saveName = " + saveName)
    
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    # Get homebrewed video sequences and corresponding masks
    print("Reading augmented image and mask sequences")
    checkpointSaveDir = "./savedNetworks/" + saveName
    # Epoch parameters
    peekEveryNEpochs=25
    saveEveryNEpochs=25
    nEpochs = 20000
    batchSize = 512
    
    
    # Complete datasets
    x_set1, y_set1, id_set1 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48","augMask_"))
    id_set1_plus = [id+"_01" for id in id_set1]
    x_set2, y_set2, id_set2 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48","augMask_"))
    id_set2_plus = [id+"_02" for id in id_set2]
    
    x_set3, y_set3, id_set3 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48","augMask_"))
    id_set3_plus = [id+"_03" for id in id_set3]
    x_set4, y_set4, id_set4 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48","augMask_"))
    id_set4_plus = [id+"_04" for id in id_set4]
    x_set5, y_set5, id_set5 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_baby","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_baby","augMask_"))
    id_set5_plus = [id+"_05" for id in id_set5]
    x_set6, y_set6, id_set6 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_baby","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_baby","augMask_"))
    id_set6_plus = [id+"_06" for id in id_set6]
    x_set7, y_set7, id_set7 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_baby","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_baby","augMask_"))
    id_set7_plus = [id+"_07" for id in id_set7]
    x_set8, y_set8, id_set8 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_baby","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_baby","augMask_"))
    id_set8_plus = [id+"_08" for id in id_set8]
    
    x_all = np.concatenate([x_set1,x_set2,x_set3,x_set4,x_set5,x_set6,x_set7,x_set8],axis=0)
    y_all = np.concatenate([y_set1,y_set2,y_set3,y_set4,y_set5,y_set6,y_set7,y_set8],axis=0)
    id_all = np.concatenate([id_set1,id_set2,id_set3,id_set4,id_set5,id_set6,id_set7,id_set8],axis=0)
    id_all_plus = np.concatenate([id_set1_plus,id_set2_plus,id_set3_plus,id_set4_plus,id_set5_plus,id_set6_plus,id_set7_plus,id_set8_plus],axis=0)
    
    '''
    x_all = np.concatenate([x_set1,x_set2])
    y_all = np.concatenate([y_set1,y_set2])
    id_all = np.concatenate([id_set1,id_set2])
    id_all_plus = np.concatenate([id_set1_plus,id_set2_plus])
    '''
    '''
    # Smaller datasets for faster debugging    
    x_all, y_all, id_all = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48","augMask_"))
    id_all_plus = [id+"_01" for id in id_all]
    '''
    '''
    x_all, y_all, id_all = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","sortTest","augImage_"),
        os.path.join("augmentedContinuousSequences","sortTest","augMask_"))
    '''    
    
    
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
    
    '''
    # Assemble each siamese arm. This is the 1-frame delta temporal siamese version.
    x_trainA        = x_train       [:,:-1,:,:] # extract all but last temporal frame
    y_train_pmMaskA = y_train_pmMask[:,:-1,:,:] # extract all but last temporal frame
    x_testA         = x_test        [:,:-1,:,:] # extract all but last temporal frame
    y_test_pmMaskA  = y_test_pmMask [:,:-1,:,:] # extract all but last temporal frame
    x_trainB        = x_train       [:,1: ,:,:] # extract all but first temporal frame
    y_train_pmMaskB = y_train_pmMask[:,1: ,:,:] # extract all but first temporal frame
    x_testB         = x_test        [:,1: ,:,:] # extract all but first temporal frame
    y_test_pmMaskB  = y_test_pmMask [:,1: ,:,:] # extract all but first temporal frame
    
    # Flatten the first two dimensions of everything (augmentation and temporal frame)
    x_trainA        = x_trainA.reshape       (-1, *x_trainA.shape[-2:])
    y_train_pmMaskA = y_train_pmMaskA.reshape(-1, *y_train_pmMaskA.shape[-2:])
    x_testA         = x_testA.reshape        (-1, *x_testA.shape[-2:])
    y_test_pmMaskA  = y_test_pmMaskA.reshape (-1, *y_test_pmMaskA.shape[-2:])
    x_trainB        = x_trainB.reshape       (-1, *x_trainB.shape[-2:])
    y_train_pmMaskB = y_train_pmMaskB.reshape(-1, *y_train_pmMaskB.shape[-2:])
    x_testB         = x_testB.reshape        (-1, *x_testB.shape[-2:])
    y_test_pmMaskB  = y_test_pmMaskB.reshape (-1, *y_test_pmMaskB.shape[-2:])
    '''
    
    # Run the complete training on the hourglass neural net
    print("Running siamese hourglass training")
    heatmaps = train_siamese_hourglass_nn(
            x_trainA, y_train_pmMaskA, x_testA, y_test_pmMaskA, 
            x_trainB, y_train_pmMaskB, x_testB, y_test_pmMaskB, 
        checkpointSaveDir = checkpointSaveDir, peekEveryNEpochs = peekEveryNEpochs,
        saveEveryNEpochs=saveEveryNEpochs, nEpochs=nEpochs, batchSize=batchSize,
        siameseWeight=siameseWeight, firstMomentWeight=firstMomentWeight, secondMomentWeight=secondMomentWeight)
        
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
    
        
