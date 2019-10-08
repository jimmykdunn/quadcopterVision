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

tf.logging.set_verbosity(tf.logging.INFO)

"""
Extract a batch of data of the input length from data x and truth y. Epoch
is input to ensure that we extract every image over the course of many
consecutive epochs.
INPUTS: 
    length: number of images (datapoints) in each batch
    x: full set of training images, [numImages,width,length]
    y_mask: full set of mask images, [numImages,width,length]
    epcoh: training epoch, integer starting at zero. Used to prevent overlap
    OPTIONAL INPUTS:
        randomDraw: set to true to extract batches randomly (default False)
RETURNS:
    [x_batch, y_batch] subsets of the x and y inputs of length length. 
"""
def extractBatch(length, x, y_mask, epoch, randomDraw=False):
    
    # Just pull them in order, wrapping to the beginning when we go over
    while (epoch+1)*length > len(x):
        epoch -= int(len(x)/length)
    
    if not randomDraw:
        # Pull in order
        batch = [epoch*length + i for i in range(length)]
    else:
        # Extract random set of prescribed length
        batch = [i for i in range(len(x))]
        random.shuffle(batch)
        batch = batch[:length]
    
    x_batch = x[batch,:,:]
    y_batch = y_mask[batch,:,:]
    
    return x_batch, y_batch


"""
Constructs a weight variable with the given shape (random initialization)
INPUTS:
     shape: N-D vector with size of random weights to generate. 
            [[weightSize], layerSize]
EXAMPLE:
    weights = weight_variable([5,5,1,32])
    Variable for the initial weights for a 5x5 convolution on a single color
    image with layer size 32.
RETURNS:
    Normally distributed tensorflow variable with dimensions shape
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

"""
Constructs a bias variable with the given shape (constant initialization)
INPUTS:
    shape: size of random biases to generate [(integer)]
EXAMPLE:
    biases = bias_variable([1024])
    Variable for the initial biases for a layer with size 1024
RETURNS:
    Constant tensorflow variable with dimensions shape
"""
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

"""
Wrapper for tensorflow 2D convolution.
INPUTS:
    x: input to the convolutional layer. Tensor [-1,width,height,nChannels]
    W: random weights to use. Tensor [nx,ny,nChannels,layerSize]
EXAMPLE:
    output = conv2d(x,W)
RETURNS:
    Result of the convolutional layer with full stride and same size
"""
def conv2d(x, W):
  stride = [1,1,1,1] # stride of filter.  Full stride means 1
  padding = 'SAME' # full zeropadding
  return tf.nn.conv2d(x,W,stride,padding)

"""
Wrapper for tensorflow maxpooling.
INPUTS:
    x: input to the maxpooling layer. Tensor [-1,width,height,nChannels]
    kernelSize: size of kernel.  Forced to be square.
EXAMPLE:
    output = max_pool(x,W)
RETURNS:
    Result of the maxpooling layer - half the widht and height of input x.
    Full-stride.
"""
def max_pool(x,kernelSize):
  ksize = [1,kernelSize,kernelSize,1]
  strides = [1,kernelSize,kernelSize,1]
  padding = 'SAME'
  return tf.nn.max_pool(x,ksize,strides,padding)


"""
Runs the upconvolution layer used in an hourglass network.  This operation is
done by taking a lower-resolution image and upsampling, then running a usual 
2D convolution operation.  This is generally followed with a concatenation by 
the layer earlier in the hourglass CNN with the same size.
INPUTS:
    x: lower-resolution layer of the CNN, size [-1,nx,ny,nChan]
    W: weights (convolution kernels), size [nkx,nky,nIn,nOut]
EXAMPLE:
    xHigh = upconv2d(x,W)
RETURNS:
    Larger resolution layer after upconvolution
"""
def upconv2d(x, wShape):    
    # Build the upconvolution layer
    with tf.name_scope('myConv2dTranspose'):
        convStride = [wShape[0]-1,wShape[1]-1]
        # inputs, filters, kernelsize, stride
        xUpconv = tf.layers.conv2d_transpose(x, filters=wShape[2], 
            kernel_size=tuple(wShape[:2]), strides=tuple(convStride), padding='SAME')
    
    # good example code:
    # https://riptutorial.com/tensorflow/example/29767/using-tf-nn-conv2d-transpose-for-arbitary-batch-sizes-and-with-automatic-output-shape-calculation-
    
    return xUpconv


   
"""
Adds a skip connection from xOld to xNew - this is a simple concatenation of
xOld and xNew. Used for hourglass CNN's
INPUTS:
    xNew: More-recent layer in the CNN. Size [?,nx.ny,nc]
    xOld: Older layer in the CNN to append to xNew. Size [?,nx,ny,nc]
EXAMPLE:
    xWithSkipConn = addSkipConnection(xNew,xOld)
RETURNS:
    Catenation of the new and old layers
"""
def addSkipConnection(xNew, xOld): 
    return tf.concat([xNew, xOld], 3)


"""
Takes in the entire list of images and creates boolean masks from them.
The boolean masks represent a segmentation of "pixels drawn on" (true) versus
"pixels not drawn on" (false). Done by a simple thresholding of the normalized 
[0,1] images. Designed for NMIST handwritten digit dataset, but should work on
anything.
INPUTS:
    images: images to threshold to get a binary mask. [-1,nx,ny], range [0,1]
    (optional) threshold: True/False threshold value. Default 0.5
EXAMPLE:
    booleanMasks = makeThresholdMask(images)
RETURNS:
    Boolean (true,false) masks for each image in the input set
"""
def makeThresholdMask(images, threshold = 0.5): 
    return images > threshold



"""
Takes a boolean mask and makes the True pixels +1 and the False pixels -1.
Values to replace true and false pixels by are overrideable with keywords.
Ratio of |falseVal| to trueVal should be approximately the expected ratio of 
target pixels to non-target pixels in imagery to maintain a balanced PD/PFa.
INPUTS:
    masks: boolean masks to threshold to get a binary mask. [-1,nx,ny]
    trueVal  (optional): value to repalce true  pixels by. Default +1
    falseVal (optional): value to repalce false pixels by. Default -0.01
EXAMPLE:
    plusMinusMask = booleanMaskToPlusMinus(booleanMasks)
RETURNS:
    Floating point masks for each boolean mask in the input set
"""
def booleanMaskToPlusMinus(booleanMask, trueVal=1, falseVal=-0.01):
    plusMinusMask = np.zeros(booleanMask.shape)
    plusMinusMask[booleanMask] = trueVal
    plusMinusMask[np.logical_not(booleanMask)] = falseVal
    
    return plusMinusMask



"""
Build the hourglass NN template.  This is a 3-hidden layer convolutional neural
network. It has 2 layers of convolution+pooling, followed by two feed-forward
(dense) layers.
INPUTS:
    x: input images, [-1,28,28,1] (-1 is for the number of images in the batch)
EXAMPLE:
    heatmap = hourglass_nn(x)
RETURNS:
    heatmap: map of pixel values, higher for pixles more likely to be target.
             Same size as input x.
"""
def hourglass_nn(x):
    
    nBatch, nWidth, nHeight = x.shape[:3]
    
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x,[-1,nWidth,nHeight,1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('firstConv'):
        w1 = weight_variable([5,5,1,32])
        #w1 = weight_variable([7,7,1,32])
        h_conv1 = tf.nn.relu(conv2d(x_image,w1)) # [-1,28,28,32]

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('firstPool'):
        #h_pool1 = max_pool(h_conv1,2) # [-1,14,14,32]
        h_pool1 = max_pool(h_conv1,4)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('secondConv'):
        w2 = weight_variable([5,5,32,64]) 
        #w2 = weight_variable([7,7,32,64]) 
        h_conv2 = tf.nn.relu(conv2d(h_pool1,w2)) # [-1,14,14,64]

    # Second pooling layer.
    with tf.name_scope('secondPool'):
        h_pool2 = max_pool(h_conv2,2) # [-1,7,7,64]  
        #h_pool2 = max_pool(h_conv2,4) # [-1,7,7,64]  

    # Remember the order is skip-connection THEN upconv
    # x_image shape is [-1,28,28,1]
    # h_conv1 shape is [-1,28,28,32]
    # h_pool1 shape is [-1,14,14,32] 
    # h_conv2 shape is [-1,14,14,64]
    # h_pool2 shape is [-1,7,7,64]   

    with tf.name_scope('secondUpconv'):
        # No skip connection necessary on the innermost layer
        h_upconv1 = tf.nn.relu(upconv2d(h_pool2, [3,3,32,64])) # [-1,14,14,32]
        
    with tf.name_scope('firstUpconv'):
        h_sk1 = addSkipConnection(h_upconv1, h_pool1) # skip connection [-1,14,14,64]
        heatmaps = tf.nn.relu(upconv2d(h_sk1, [5,5,1,64])) # [-1,28,28,1]
        
    # The size of heatmap here should be [batch,28,28,1] for NMIST
    return heatmaps
    

# end hourglass_nn


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
        b_heatmaps = hourglass_nn(b_images)
        b_heatmaps = tf.reshape(b_heatmaps,[-1,nWidth,nHeight],'b_heatmaps')
   
    # The heatmap loss calculation
    with tf.name_scope('heatmapGain'):
        # Gain here is really just an pixel-wise heatmap*truthmask product
        # + gain for every heatmap pixel that IS     part of the targetmask
        # - gain for every heatmap pixel that IS NOT part of the targetmask
        # To do this, targetmask must have +1's at target     locations
        # and         targetmask must have -1's at background locations
        # Make sure targetmask is formed in this way!!!
        b_gainmaps = tf.multiply(b_heatmaps, b_masks) # pixel-by-pixel gain
        b_gainmaps = tf.math.minimum(b_gainmaps, 1.0, name="b_gainmaps") # anything above 1 doesn't help
        
        # May be useful to have an intermediate reduction here of a single
        # gain value for each individual image...
        
        # Average of gain across every pixel of every image
        gain = tf.reduce_mean(tf.cast(b_gainmaps,tf.float32))
        loss = tf.multiply(-1.0,gain)
        
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
            batch = extractBatch(batchSize, trainImages, trainMasks, epoch)
            
            # Run a single epoch with the extracted data batch
            train_step.run(feed_dict={b_images: batch[0], b_masks: batch[1]}) 
            
            # Check our progress on the training data every peekEveryNEpochs epochs
            if epoch % peekEveryNEpochs == (peekEveryNEpochs-1):
                trainGain = gain.eval(feed_dict={b_images: batch[0], b_masks: batch[1]})
                perfectTrainGain = perfectGain.eval(feed_dict={b_images: batch[0], b_masks: batch[1]})
                print('epoch %d of %d, training gain %g' % (epoch+1, nEpochs, trainGain/perfectTrainGain))
                testBatch = extractBatch(100, testImages, testMasks, 0, randomDraw=True)
                testGain = gain.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                perfectTestGain = perfectGain.eval(feed_dict={b_images: testBatch[0], b_masks: testBatch[1]})
                print('epoch %d of %d, test gain %g' % (epoch+1, nEpochs, testGain/perfectTestGain))
                peekSchedule.append(epoch+1)
                trainGainHistory.append(trainGain/perfectTrainGain)
                testGainHistory.append(testGain/perfectTestGain)
            
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
        save_graph_protobuf(sess,checkpointSaveDir)
        
        print("\n\n\n\n")
        print("============================")
        print("TRAINING RESULTS")
        print("============================")

        # Total elapsed time
        end_sec = time.clock()
        print('Total elapsed time for %d epochs: %g sec' 
              % (nEpochs, end_sec-start_sec))
    
        # Finish off by running the test set.  Extract the entire test set.
        test_batch = extractBatch(len(testImages), testImages, testMasks, 0)
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
save_graph_protobuf()
    Saves a checkpoint file and protobuf files containing the graph (structure)
    of a forward pass and the trained weights.  All of that is contained in the
    sess variable.  The resulting files can be loaded at a later time and new
    never-before-seen images can be run through them using the 
    use_hourglass_cnn*.py scripts.
INPUTS:
    sess: tensorflow session to save
    directory: base directory to save to.  Will have files added to it.
OPTIONAL INPUTS:
    baseName: name of saved files (without extensions), (default "modelFinal")
EXAMPLE:
    test_heatmaps = save_graph_protobuf(sess,'myTrainedModel')
RETURNS:
    Saves the graph to the directory+baseName appended with ".ckpt", ".pb", 
    and ".pbtxt"

Based on: https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/
"""
def save_graph_protobuf(sess,directory,baseName='modelFinal'):
    # Save check point for graph frozen later.  By itself this could actually
    # be used to reload and run a forward pass of the model, but it is not
    # compatible with cv2.
    saver = tf.train.Saver(max_to_keep=100)
    ckpt_filepath = os.path.join(directory,baseName+'.ckpt')
    saver.save(sess, ckpt_filepath)
    
    # Convert to protobuf with the utility function
    tf_utils.ckpt_to_protobuf(ckpt_filepath)
# end save_graph_protobuf   
    

"""
Build and train the hourglass CNN from the main level when this file is called.
"""

if __name__ == "__main__":  
    
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    '''
    # Get the MNIST handwritten-digit data
    x_train, y_train, x_test, y_test = mnist.getMNISTData()
    checkpointSaveDir = "./mnist_hourglass_nn_save";
    # Make MNIST mask truth by simple thresholding
    y_train_pmMask = booleanMaskToPlusMinus(makeThresholdMask(x_train),trueVal=1,falseVal=-0.1)
    y_test_pmMask  = booleanMaskToPlusMinus(makeThresholdMask(x_test), trueVal=1,falseVal=-0.1)
    peekEveryNEpochs=50
    saveEveryNEpochs=100
    nEpochs = 100
    '''
    
    # Get homebrewed video sequences and corresponding masks
    print("Reading augmented image and mask sequences")
    checkpointSaveDir = "./homebrew_hourglass_nn_save";
    # Epoch parameters
    peekEveryNEpochs=25
    saveEveryNEpochs=25
    nEpochs = 1000
    batchSize = 512
    '''
    x_set1, y_set1, idSet1 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_PHO_hallway_64x64","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_PHO_hallway_64x64","augMask_"))
    x_set2, y_set2, idSet2 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_BOS_trainSidewalk_64x64","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_BOS_trainSidewalk_64x64","augMask_"))
    '''
    
    x_set1, y_set1, id_set1 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x64","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x64","augMask_"))
    x_set2, y_set2, id_set2 = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab2_64x64","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab2_64x64","augMask_"))
    x_all = np.concatenate([x_set1,x_set2],axis=0)
    y_all = np.concatenate([y_set1,y_set2],axis=0)
    id_all = np.concatenate([id_set1,id_set2],axis=0)
    
    '''
    x_all, y_all, id_all = vu.pull_aug_sequence(
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x64_mini","augImage_"),
        os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x64_mini","augMask_"))
    '''
    
    # Split into train and test sets randomly
    #x_train, y_train, x_test, y_test = \
    #    vu.train_test_split(x_all, y_all, trainFraction=0.8)
    x_train, y_train, x_test, y_test = \
        vu.train_test_split_noCheat(x_all, y_all, id_all, trainFraction=0.8)

    # Convert masks to appropriately-weighted +/- masks
    y_train_pmMask = booleanMaskToPlusMinus(y_train)
    y_test_pmMask  = booleanMaskToPlusMinus(y_test)
    
    
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
    
        
