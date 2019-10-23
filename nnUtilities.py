# -*- coding: utf-8 -*-
"""
FILE: nnUtilities.py
DESCRIPTION:
    Neural network utility functions

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""
import tensorflow as tf
import numpy as np
import random
import tf_utils
import os

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

