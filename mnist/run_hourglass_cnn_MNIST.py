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
import matplotlib.pyplot as plt
import time
import cv2
import copy

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
RETURNS:
    [x_batch, y_batch] subsets of the x and y inputs of length length. 
"""
def extractBatch(length, x, y_mask, epoch):
    
    # Just pull them in order, wrapping to the beginning when we go over
    while (epoch+1)*length > len(x):
        epoch -= int(len(x)/length)
    
    batch = [epoch*length + i for i in range(length)]
    
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
Wrapper for tensorflow 2x2 maxpooling.
INPUTS:
    x: input to the maxpooling layer. Tensor [-1,width,height,nChannels]
EXAMPLE:
    output = max_pool_2x2(x,W)
RETURNS:
    Result of the maxpooling layer - half the widht and height of input x.
    Full-stride.
"""
def max_pool_2x2(x):
  ksize = [1,2,2,1]
  strides = [1,2,2,1]
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
def upconv2d(x, W):
    # It is VERY IMPORTANT to get these variables in this way
    nx = x.get_shape().as_list()[1]
    ny = x.get_shape().as_list()[2]
    nkx = W.get_shape().as_list()[0] 
    nky = W.get_shape().as_list()[1] 
    nW = W.get_shape().as_list()[2] 
    
    # Output dimension and stride calculations
    # use tf.shape(x)[0] instead of -1
    outShape = tf.stack([tf.shape(x)[0], nkx*nx, nky*ny, nW]) 
    stride = [1,nkx,nky,1]
    
    # Build the upconvolution layer
    xUpconv = tf.nn.conv2d_transpose(x, W, outShape, stride, padding='SAME') 
    
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
INPUTS:
    masks: boolean masks to threshold to get a binary mask. [-1,nx,ny]
    trueVal  (optional): value to repalce true  pixels by. Default +1
    falseVal (optional): value to repalce false pixels by. Default -1
EXAMPLE:
    plusMinusMask = booleanMaskToPlusMinus(booleanMasks)
RETURNS:
    Floating point masks for each boolean mask in the input set
"""
def booleanMaskToPlusMinus(booleanMask, trueVal=1, falseVal=-1):
    plusMinusMask = np.zeros(booleanMask.shape)
    plusMinusMask[booleanMask] = trueVal
    plusMinusMask[np.logical_not(booleanMask)] = falseVal
    
    return plusMinusMask


"""
Loads the NMIST handwritten-digits data using the built-in keras function.
Performs normalization and also displays a few example images if python 
terminal allows it.
INPUTS:
    None
EXAMPLE:
    x_train, y_train, x_test, y_test = getNMISTData()
RETURNS:
    x_train: full set of 60000 28x28 pixel training images, [60000,28,28]
    y_train: full set of 60000 truth classes for x_train [60000]
    x_test: full set of 10000 28x28 pixel test images, [10000,28,28]
    y_test: full set of 10000 truth classes for x_test [10000]
"""
def getNMISTData():
    """Load training and eval data"""
    ((x_train, y_train),
     (x_test, y_test)) = tf.keras.datasets.mnist.load_data()
    print("Number of training images: " + str(x_train.shape[0]))
    print("Number of testing  images: " + str(x_test.shape[0]))
    print("Image shape (nx,ny): (" + str(x_train.shape[1]) + 
                               "," + str(x_train.shape[2]) + ")")
    
    x_train = x_train/np.float32(255) # normalize images
    y_train = y_train.astype(np.int32)  # not required

    x_test = x_test/np.float32(255) # normalize images
    y_test = y_test.astype(np.int32)  # not required
    
    # Display some example data
    chain = np.squeeze(x_train[0,:,:])
    for i in range(15):
        chain = np.append(chain,np.squeeze(x_train[i+1,:,:]),axis=1)
    print("Example training data")
    plt.figure()
    plt.imshow(chain)
    labelstrs = "".join([str(truth) + ", " for truth in y_train[:16]])
    print("Truth: ", labelstrs)
    
    return x_train, y_train, x_test, y_test

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
    
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x,[-1,28,28,1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        w1 = weight_variable([5,5,1,32])
        h_conv1 = tf.nn.relu(conv2d(x_image,w1)) # [-1,28,28,32]

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1) # [-1,14,14,32]

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        w2 = weight_variable([5,5,32,64]) 
        h_conv2 = tf.nn.relu(conv2d(h_pool1,w2)) # [-1,14,14,64]

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2) # [-1,7,7,64]  

    # Remember the order is skip-connection THEN upconv
    # x_image shape is [-1,28,28,1]
    # h_conv1 shape is [-1,28,28,32]
    # h_pool1 shape is [-1,14,14,32] 
    # h_conv2 shape is [-1,14,14,64]
    # h_pool2 shape is [-1,7,7,64]   

    with tf.name_scope('upconv2'):
        # No skip connection necessary on the innermost layer
        wu2 = weight_variable([2,2,32,64])
        h_upconv1 = tf.nn.relu(upconv2d(h_pool2, wu2)) # [-1,14,14,32]
        
    with tf.name_scope('upconv1'):
        h_sk1 = addSkipConnection(h_upconv1, h_pool1) # skip connection [-1,14,14,64]
        wu1 = weight_variable([2,2,1,64])
        heatmap = tf.nn.relu(upconv2d(h_sk1, wu1)) # [-1,28,28,1]
        
    # The size of heatmap here should be [batch,28,28,1] for NMIST
    return heatmap
    

# end hourglass_nn

"""
Build and train the hourglass CNN from the main level when this file is called.
"""

if __name__ == "__main__":  
    
    # Clear checkpoint files to get a clean training run each time
    checkpointSaveDir = "./mnist_hourglass_nn_save" # checkpoints saved here
    if os.path.exists(checkpointSaveDir): # only rm if it exists
        shutil.rmtree(checkpointSaveDir, ignore_errors=True)    
    
    # Get the NMIST handwritten-digit data
    x_train, y_train, x_test, y_test = getNMISTData()
    
    # Make mask truth by sim0ple thresholding
    y_train_pmMask = booleanMaskToPlusMinus(makeThresholdMask(x_train))
    y_test_pmMask  = booleanMaskToPlusMinus(makeThresholdMask(x_test))
    

    # Placeholders for the data and associated truth
    x = tf.placeholder(tf.float32, [None, 28,28], name="x")
    y_pmMask = tf.placeholder(tf.float32, [None, 28,28], name="y_pmMask")
    
    # Build the graph for the deep hourglass net
    # It is best to literally thing of this as just building the graph, since
    # it is in reality just a "placeholder" or template for what we will
    # actually be running. y_conv is [-1,10], where -1 is the number of input
    # datapoints and 10 is the probability (logit) for each output class 
    # (numeral).
    heatmap = hourglass_nn(x)
   
    # The heatmap loss calculation
    with tf.name_scope('heatmapGain'):
        # Gain here is really just an pixel-wise heatmap*truthmask product
        # + gain for every heatmap pixel that IS     part of the targetmask
        # - gain for every heatmap pixel that IS NOT part of the targetmask
        # To do this, targetmask must have +1's at target     locations
        # and         targetmask must have -1's at background locations
        # Make sure targetmask is formed in this way!!!
        gainmap = tf.multiply(tf.reshape(heatmap,[-1,28,28]), y_pmMask) # pixel-by-pixel gain
        gainmap = tf.math.minimum(gainmap, 1.0) # anything above 1 doesn't help
        
        # May be useful to have an intermediate reduction here of a single
        # gain value for each individual image...
        
        # Average of gain across every pixel of every image
        gain = tf.reduce_mean(tf.cast(gainmap,tf.float32))
        loss = tf.multiply(-1.0,gain)
        
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
    saver = tf.train.Saver()
    
    # Start the clock
    start_sec = time.clock()
    
    # Actually execute the training using the CNN template we just built
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Loop over every epoch
        nEpochs = 100
        for epoch in range(nEpochs): 
            # Extract data for this batch
            batch = extractBatch(100, x_train, y_train_pmMask, epoch)
            
            # Run a single epoch with the extracted data batch
            train_step.run(feed_dict={x: batch[0], y_pmMask: batch[1]}) 
            
            # Check our progress on the training data every N epochs
            if epoch % 50 == 49:
                trainGain = gain.eval(feed_dict={x: batch[0], y_pmMask: batch[1]})
                print('epoch %d of %d, training gain %g' % (epoch+1, nEpochs, trainGain))
            
            # Print elapsed time every 100 epochs
            if epoch % 50 == 49:
                curr_sec = time.clock()
                print('    Elapsed time for %d epochs: %g sec' 
                      % (epoch+1, curr_sec-start_sec))
    
            # Save the model weights (and everything else) every 500 epochs
            if epoch % 500 == 499 or epoch == (nEpochs-1):
                save_path = saver.save(sess, checkpointSaveDir + "/model_at" + str(epoch+1) + ".ckpt")
                print("    Checkpoint saved to: %s" % save_path)
        
        print("\n\n\n\n")
        print("============================")
        print("TRAINING RESULTS")
        print("============================")

        # Total elapsed time
        end_sec = time.clock()
        print('Total elapsed time for %d epochs: %g sec' 
              % (nEpochs, end_sec-start_sec))
    
        # Finish off by running the test set.  Extract the entire test set.
        test_batch = extractBatch(len(x_test), x_test, y_test_pmMask, 0)
        test_gain = gain.eval(feed_dict={x: test_batch[0], y_pmMask: test_batch[1]})
        test_heatmaps = heatmap.eval(feed_dict={x: test_batch[0], y_pmMask: test_batch[1]})
        print('test gain %g' % test_gain)
        
        # Write out the first few heatmaps to file along with the associated
        # test data inputs for visualization
        if not os.path.isdir('heatmaps'): # make the output dir if needed
            os.mkdir('heatmaps')
        numToWrite = np.min([10,test_heatmaps.shape[0]])
        filmstrip = []
        for iHeat in range(numToWrite):
            # Make the output images individually
            heatmapOutArray = np.squeeze(test_heatmaps[iHeat,:])*255.0
            testOutArray = np.squeeze(x_test[iHeat,:])*255.0
            
            # Join heatmap and actual image to a single array for output
            joinedStr = 'joined_%04d.png' % iHeat
            joined = np.concatenate([testOutArray, heatmapOutArray],axis=0)
            cv2.imwrite(os.path.join('heatmaps',joinedStr), joined)
            print('Wrote ' + os.path.join('heatmaps',joinedStr))
            
            # Make output strip of images and heatmaps
            if iHeat == 0:
                filmstrip = copy.deepcopy(joined)
            filmstrip = np.concatenate([filmstrip,joined], axis=1)
            
        # Write all numToWrite in a single image for easy analysis
        cv2.imwrite(os.path.join('heatmaps','filmstrip.png'), filmstrip) 
        print('Wrote ' + os.path.join('heatmaps','filmstrip.png')) 
        
        # Print the location of the saved network
        print("Final trained network saved to: " + save_path)
        print("You can use use_cnn.py with this final network to classify new datapoints")
        
