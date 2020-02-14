# -*- coding: utf-8 -*-
"""
FILE: train_siamese_hourglass_cnn.py
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
import matplotlib.pyplot as plt
import nnUtilities as nnu
import neuralNetStructures as nns

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
def train_siamese_hourglass_cnn(trainImagesA, trainMasksA, testImagesA, testMasksA, \
                               trainImagesB, trainMasksB, testImagesB, testMasksB, \
    nEpochs=100, batchSize=100, checkpointSaveDir='./hourglass_nn_save', \
    saveEveryNEpochs=500, peekEveryNEpochs=50, siameseWeight = 1.0,
    firstMomentWeight = 1.0, secondMomentWeight = 1.0, continuePrevious=True):
    
    print("BEGIN SIAMESE HOURGLASS NN TRAINING")
    
    # Clear checkpoint files to get a clean training run each time
    if os.path.exists(checkpointSaveDir): 
        if continuePrevious: # continue training from previously saved network
            pass
        else: # remove existing saves
            shutil.rmtree(checkpointSaveDir, ignore_errors=True) 
    else:
        continuePrevious = False
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
        # Add if statements for NaN stability and processing speed when one
        # or more of the weights is set to zero.
        loss = heatmapLoss
        if siameseWeight != 0.0:
            loss = tf.add(loss,tf.multiply(siameseWeight, siameseLoss))
        if firstMomentWeight != 0.0:
            loss = tf.add(loss,tf.multiply(firstMomentWeight, firstMomentLoss))
        if secondMomentWeight != 0.0:
            loss = tf.add(loss,tf.multiply(secondMomentWeight, secondMomentLoss))
                
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
        
        # Load earlier session if we are continuing from a previous one
        if continuePrevious: # continue training from previously saved network
            # Determine the latest epoch saved in checkpointSaveDir
            files = os.listdir(checkpointSaveDir)
            lastEpoch = 0
            for cFile in files:
                if cFile[:8] == 'model_at':
                    dotPos = cFile.find('.')
                    epochNum = int(cFile[8:dotPos])
                    if epochNum > lastEpoch:
                        lastEpoch = epochNum
            
            # Restore it
            checkpointSaveFile = checkpointSaveDir + "/model_at" + str(lastEpoch) + ".ckpt"
            print("Restoring graph from %s and continuing training" % checkpointSaveFile)
            saver.restore(sess,checkpointSaveFile)
            print("Successfully restored saved session")
            nEpochs -= lastEpoch
            epochsToRun = np.arange(nEpochs) + lastEpoch
        else:
            epochsToRun = np.arange(nEpochs)
    
        
        # Loop over every epoch
        peekSchedule = []
        trainGainHistory = []
        testGainHistory = []
        for epoch in epochsToRun: 
        #for epoch in range(nEpochs): 
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
            #if epoch % saveEveryNEpochs == (saveEveryNEpochs-1) or epoch == (nEpochs-1):
            if epoch % saveEveryNEpochs == (saveEveryNEpochs-1) or epoch == (nEpochs-1) or epoch == epochsToRun[0]:
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
    
