# -*- coding: utf-8 -*-
"""
FILE: run_hourglass_cnn.py
DESCRIPTION:
    Trains and evaluates a convolutional neural net (CNN) to generate a heatmap
    of the probability of a quadcopter being at each pixel in the input image.
    Uses tensorflow.  
    
    Based on NMIST homework from CS542 at Boston University in Fall 2018

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
import tensorflow as tf
import numpy as np
import os
import shutil
import mnist
import time
import cv2
import copy
import videoUtilities as vu
import matplotlib.pyplot as plt
import random
import tf_utils
import nnUtilities as nnu
import neuralNetStructures as nns

tf.logging.set_verbosity(tf.logging.INFO)


"""
Train the hourglass NN. Makes the graph with hourglass_nn, then calculates gain
and optimizes in batch over the input data. Ends with a test forward pass.
"-1" is a placeholder for the number of images in the batch.
INPUTS:
    trainImages: input images, [-1,nx,ny,1] 
    trainMasks: truth masks associated with trainImages. +1 if target, -1 otherwise, [-1,nx,ny] 
    testImages: test images, [-1,nx,ny,1]
    testMasks: truth masks associated with testImages. +1 if target, -1 otherwise, [-1,nx,ny] 
OPTIONAL INPUTS:
    nEpochs: number of training epcohs to run (default 100)
    batchSize: number of images to use per training epoch (default 100)
    checkpointSaveDir: directory to save trained neural net and graph to (default './hourglass_nn_save")
    saveEveryNEpochs: save checkpoint every this many epochs (and at the end) (default 500)
    peekEveryNEpochs: print the training gain with a forward pass every this many epochs (default 50)
EXAMPLE:
    test_heatmaps = train_hourglass_nn(x)
RETURNS:
    test_heatmaps: map of pixel values for each image, higher for pixels more 
                   likely to be target. Same size as input testImages.
"""
def train_hourglass_nn(trainImages, trainMasks, testImages, testMasks, \
    nEpochs=100, batchSize=100, checkpointSaveDir='./hourglass_nn_save', \
    saveEveryNEpochs=500, peekEveryNEpochs=50):
    
    print("BEGIN HOURGLASS NN TRAINING")
    
    # Clear checkpoint files to get a clean training run each time
    if os.path.exists(checkpointSaveDir): # only rm if it exists
        shutil.rmtree(checkpointSaveDir, ignore_errors=True)   
    else:
        os.mkdir(checkpointSaveDir)
        
    # Image sizes
    nBatch, nWidth, nHeight = trainImages.shape[:3]

    # Placeholders for the data and associated truth
    # "b_" prefix stands for "batch"
    with tf.name_scope('inputs'):
        b_images = tf.placeholder(tf.float32, [None, nWidth,nHeight], name="b_images")
        b_masks = tf.placeholder(tf.float32, [None, nWidth,nHeight], name="b_masks")
    
    # Build the graph for the deep hourglass net
    # It is best to literally thing of this as just building the graph, since
    # it is in reality just a "placeholder" or template for what we will
    # actually be running. y_conv is [-1,10], where -1 is the number of input
    # datapoints and 10 is the probability (logit) for each output class 
    # (numeral).
    with tf.name_scope('heatmaps'):
        b_heatmaps = nns.hourglass_nn(b_images)
        b_heatmaps = tf.reshape(b_heatmaps,[-1,nWidth,nHeight],'b_heatmaps')
   
    # The heatmap loss calculation
    with tf.name_scope('heatmapGain'):
        # Gain here is really just an pixel-wise heatmap*truthmask product
        # + gain for every heatmap pixel that IS     part of the targetmask
        # - gain for every heatmap pixel that IS NOT part of the targetmask
        # To do this, targetmask must have +1's at target     locations
        # and         targetmask must have -1's at background locations
        # Make sure targetmask is formed in this way!!!
        
        # Force heatmap to be in the range 0 to 1
        b_heatmaps = tf.math.maximum(b_heatmaps,tf.constant(0.0))
        b_gainmaps = tf.multiply(b_heatmaps, b_masks) # pixel-by-pixel gain
        b_gainmaps = tf.math.minimum(b_gainmaps, 1.0, name="b_gainmaps") # anything above 1 doesn't help
        
        # May be useful to have an intermediate reduction here of a single
        # gain value for each individual image...
        
        # Average of gain across every pixel of every image
        gain = tf.reduce_mean(tf.cast(b_gainmaps,tf.float32))
        heatmapLoss = tf.multiply(-1.0,gain) # invert for gain -> loss
        
        # Calculate second moment loss - penalize for having heatmap energy
        # highly spread out.
        #secondMomentLoss, stdX, stdY, COM_x, COM_y, totalEnergy = \
        #    nnu.calculateSecondMomentLoss(b_heatmaps,b_masks)
        
        # Calculate the total loss
        #loss = tf.add(heatmapLoss,tf.multiply(tf.constant(0.1),secondMomentLoss))
        loss = heatmapLoss
        
        # Perfect segementation would result in this gain value
        booleanMask = tf.math.greater(b_masks,0)
        perfectGain = tf.reduce_mean(tf.cast(booleanMask,tf.float32))
        
    # Optimization calculation
    with tf.name_scope('adam_optimizer'):
        # Basic ADAM optimizer
        train_step = tf.train.AdamOptimizer().minimize(loss)
        
    # Final accuracy calculation is nothing more than the gain itself. No need
    # for an additional calculation.
        
    # Save the grap of the neural network and loss function
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
            batch = nnu.extractBatch(batchSize, trainImages, trainMasks, epoch)
            
            # Run a single epoch with the extracted data batch
            train_step.run(feed_dict={b_images: batch[0], b_masks: batch[1]}) 
            
            # Check our progress on the training data every peekEveryNEpochs epochs
            if epoch % peekEveryNEpochs == (peekEveryNEpochs-1):
                trainGain = gain.eval(feed_dict={b_images: batch[0], b_masks: batch[1]})
                perfectTrainGain = perfectGain.eval(feed_dict={b_images: batch[0], b_masks: batch[1]})
                print('epoch %d of %d, training gain %g' % (epoch+1, nEpochs, trainGain/perfectTrainGain))
                testBatch = nnu.extractBatch(100, testImages, testMasks, 0, randomDraw=True)
                testGain = gain.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                perfectTestGain = perfectGain.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                print('epoch %d of %d, test gain %g' % (epoch+1, nEpochs, testGain/perfectTestGain))
                peekSchedule.append(epoch+1)
                trainGainHistory.append(trainGain/perfectTrainGain)
                testGainHistory.append(testGain/perfectTestGain)
                '''
                # Calculate some helpful metrics
                heatmapLossE = heatmapLoss.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                secondMomentLossE = secondMomentLoss.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                stdXE = stdX.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                stdYE = stdY.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                COM_xE = COM_x.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                COM_yE = COM_y.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                totalEnergyE = totalEnergy.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                print("heatmapLoss")
                print(heatmapLossE)
                print("secondMomentLoss")
                print(secondMomentLossE)
                print("stdX")
                print(stdXE)
                print("stdY")
                print(stdYE)
                print("COM_x")
                print(COM_xE)
                print("COM_y")
                print(COM_yE)
                print("totalEnergy")
                print(totalEnergyE)
                '''
            
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
        test_batch = nnu.extractBatch(len(testImages), testImages, testMasks, 0)
        test_gain = gain.eval(feed_dict={b_images: test_batch[0], b_masks: test_batch[1]})
        perfectTestGain = perfectGain.eval(feed_dict={b_images: test_batch[0], b_masks: test_batch[1]})
        test_heatmaps = b_heatmaps.eval(feed_dict={b_images: test_batch[0], b_masks: test_batch[1]})
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
    
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    # Get homebrewed video sequences and corresponding masks
    print("Reading augmented image and mask sequences")
    checkpointSaveDir = "./homebrew_hourglass_nn_save";
    # Epoch parameters
    peekEveryNEpochs=100
    saveEveryNEpochs=100
    nEpochs = 1000 #20000
    batchSize = 512
    '''
    x_set1, y_set1, id_set1 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48","augMask_"))
    x_set2, y_set2, id_set2 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48","augMask_"))
    x_set3, y_set3, id_set3 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48","augMask_"))
    x_set4, y_set4, id_set4 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48","augMask_"))
    x_set5, y_set5, id_set5 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_baby","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_baby","augMask_"))
    x_set6, y_set6, id_set6 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_baby","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_baby","augMask_"))
    x_set7, y_set7, id_set7 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_baby","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_baby","augMask_"))
    x_set8, y_set8, id_set8 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_baby","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_baby","augMask_"))
    x_all = np.concatenate([x_set1,x_set2,x_set3,x_set4,x_set5,x_set6,x_set7,x_set8],axis=0)
    y_all = np.concatenate([y_set1,y_set2,y_set3,y_set4,y_set5,y_set6,y_set7,y_set8],axis=0)
    id_all = np.concatenate([id_set1,id_set2,id_set3,id_set4,id_set5,id_set6,id_set7,id_set8],axis=0)
    '''
    
    x_all, y_all, id_all = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48","augMask_"))
    
    
    # Split into train and test sets randomly
    #x_train, y_train, x_test, y_test = \
    #    vu.train_test_split(x_all, y_all, trainFraction=0.8)
    x_train, y_train, x_test, y_test, foo, bar = \
        vu.train_test_split_noCheat(x_all, y_all, id_all, trainFraction=0.8)

    # Convert masks to appropriately-weighted +/- masks
    y_train_pmMask = nnu.booleanMaskToPlusMinus(y_train, falseVal=-0.01)
    y_test_pmMask  = nnu.booleanMaskToPlusMinus(y_test, falseVal=-0.01)
    
    
    # Run the complete training on the hourglass neural net
    heatmaps = train_hourglass_nn(x_train, y_train_pmMask, x_test, y_test_pmMask, 
        checkpointSaveDir = checkpointSaveDir, peekEveryNEpochs = peekEveryNEpochs,
        saveEveryNEpochs=saveEveryNEpochs, nEpochs=nEpochs, batchSize=batchSize)
        
    # Write out the first few testset heatmaps to file along with the associated
    # test data inputs for visualization
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
    
        
