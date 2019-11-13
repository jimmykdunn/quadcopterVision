# -*- coding: utf-8 -*-
"""
FILE: use_hourglass_cnn
DESCRIPTION:
    Uses a trained hourglass CNN on arbitrary input image(s)
    
    Based on: https://datascience.stackexchange.com/questions/16922/using-tensorflow-model-for-prediction
    
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
import videoUtilities as vu

"""
use_hourglass_cnn()
DESCRIPTION:
    Runs a forward pass of the tensorflow graph in modelPath on input images.
    Optionally runs multiple times to get more accurate timing statistics.
    
INPUTS: 
    modelPath: checkpoint (".ckpt") file to run a forward pass with
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
            b_images = graph.get_tensor_by_name("inputs/b_images:0")
            b_masks = graph.get_tensor_by_name("inputs/b_masks:0")
            
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

"""
use_hourglass_cnn_pb()
DESCRIPTION:
    Runs a forward pass of the tensorflow graph in modelPath on input images.
    Optionally runs multiple times to get more accurate timing statistics.
    
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
def use_hourglass_cnn_pb(modelPath, inputImages, numTimingTrials=1):
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    
    with tf.gfile.GFile(modelPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
        print ('Check out the input placeholders:')
        nodes = [n.name + ' => ' + n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print('     ' + node)
        
    # Define input tensor
    b_images = tf.placeholder(np.float32, shape=[None,64,64], name='b_images')
    tf.import_graph_def(graph_def, {'inputs/b_images': b_images})
    
    # Node names for debugging
    #names = [i.name for i in sess.graph.get_operations()]
    
    # Define output tensor (heatmap)
    heatmap_tensor = graph.get_tensor_by_name("import/heatmaps/b_heatmaps:0")
    
    # Run forwad pass
    start_sec = time.clock()
    for i in range(numTimingTrials): # run many times to get timing information
        heatmaps = sess.run(heatmap_tensor, feed_dict = {b_images: inputImages})
    
      
    # Display timing trial information
    end_sec = time.clock()
    if numTimingTrials != 1:
        print("%d trials in %g seconds" % (numTimingTrials,end_sec-start_sec))
        print("Forward pass speed: %g Hz" % (numTimingTrials/(end_sec-start_sec)))
    
    
    return heatmaps

"""
twoTest()
DESCRIPTION:
    Debugging function that runs an MNIST handwritten digit CNN on a 
    handwritten "2". Intended for hourglass (segmentation) nets.
    
INPUTS: 
    modelPath: checkpoint (".ckpt") file to run a forward pass with
    
OUTPUTS: 
    Displays the resulting heatmap and saves to file

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
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
    

"""
sixTest()
DESCRIPTION:
    Debugging function that runs an MNIST handwritten digit CNN on a 
    handwritten "6". Intended for hourglass (segmentation) nets.
    
INPUTS: 
    modelPath: checkpoint (".ckpt") file to run a forward pass with
    
OUTPUTS: 
    Displays the resulting heatmap and saves to file

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
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

"""
quadcopterTest()
DESCRIPTION:
    Debugging function that runs a trained hourglass CNN on a basic quadcopter
    image.
    
INPUTS: 
    modelPath: checkpoint (".ckpt") file to run a forward pass with
    
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
    plt.imshow(joined)
    
    
"""
quadcopterBatchTest()
DESCRIPTION:
    Debugging function that runs a trained hourglass CNN on a series of basic 
    quadcopter images.
    
INPUTS: 
    modelPath: checkpoint (".ckpt") file to run a forward pass with
    
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
    modelExt = os.path.splitext(modelPath)[1]
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
                maskpoint = np.mean(maskpoint,axis=2) < 125
                print("Running CNN at " + modelPath + " on " + os.path.join(directory,filename))
                if modelExt == '.ckpt':
                    heatmap = use_hourglass_cnn(modelPath,
                        np.reshape(datapoint,[1,datapoint.shape[0],datapoint.shape[1]]),
                        numTimingTrials=100)
                elif modelExt == '.pb':
                    heatmap = use_hourglass_cnn_pb(modelPath, 
                        np.reshape(datapoint,[1,datapoint.shape[0],datapoint.shape[1]]),
                        numTimingTrials=100)
                else:
                    print('Model type not recognized: ' + modelExt)
                
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
    
    #twoTest(os.path.join('mnist_hourglass_nn_save','model_at100.ckpt'))
    #print("\n\n")
    #sixTest(os.path.join('mnist_hourglass_nn_save','model_at100.ckpt'))
    #print("\n\n")
    #quadcopterTest(os.path.join('homebrew_hourglass_nn_save','model_at750.ckpt'))
    print("\n\n")
    #quadcopterBatchTest(os.path.join('homebrew_hourglass_nn_save','model_at1000.ckpt'))
    #quadcopterBatchTest(os.path.join('homebrew_hourglass_nn_save_GOOD','model_at20000.ckpt'), directory='goldenImages')
    quadcopterBatchTest(os.path.join('savedNetworks','mirror6k_sW00p50_1M00p00_2M00p00','model_at49650.ckpt'), directory='goldenImages')
    
    
