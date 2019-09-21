# -*- coding: utf-8 -*-
"""
FUNCTION: run_cnn
DESCRIPTION:
    Trains and evaluates a convolutional neural net (CNN) to generate a heatmap
    of the probability of a quadcopter being at each pixel in the input image.
    Uses tensorflow.  
    
    Based on NMIST homework from CS542 at Boston University in Fall 2018
    
INPUTS: 

    
OUTPUTS: 
    

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
import use_cnn

tf.logging.set_verbosity(tf.logging.INFO)

# Extract a batch of data of the input length from data x and truth y. Epoch
# is input to ensure that we extract every image over the course of many
# consecutive epochs.
def extractBatch(length, x, y, epoch):
    
    # Just pull them in order, wrapping to the beginning when we go over
    while (epoch+1)*length > len(x):
        epoch -= int(len(x)/length)
    
    batch = [epoch*length + i for i in range(length)]
    
    x_batch = x[batch,:,:]
    y_batch = np.zeros([length,10]) == 1
    y_b = y[batch]
    for i in range(length):
        y_batch[i,y_b[i]] = True
    
    return x_batch, y_batch

# Constructs a weight variable with the given shape (random initialization)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Constructs a bias variable with the given shape (constant initialization)
def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Wrapper for 2D convolution
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  # Input x is 4d [image_num, x, y, chan] (chan = color) [?,28,28,1]
  # Input W (kernel) is also 4d [height, width, in chan, out chan], [28,28,1,1]
  stride = [1,1,1,1] # stride of filter.  Full stride means 1
  padding = 'SAME' # full zeropadding
  return tf.nn.conv2d(x,W,stride,padding)

# Wrapper for maxpooling
def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  # Input x is 4d [image_num, x, y, chan] (chan = color) [?,28,28,1]
  ksize = [1,2,2,1]
  strides = [1,2,2,1]
  padding = 'SAME'
  return tf.nn.max_pool(x,ksize,strides,padding)

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


# Build the CNN template.  This is a 3-hidden layer convolutional neural
# network. It has 2 layers of convolution+pooling, followed by two feed-forward
# (dense) layers.
def deepnn(x):
    """
    deepnn builds the graph for a deep net for classifying digits.
    Inputs:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x,[-1,28,28,1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        w1 = weight_variable([5,5,1,32])
        h_conv1 = tf.nn.relu(conv2d(x_image,w1))

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        w2 = weight_variable([5,5,32,64]) # [5,5,1,2]??  how is this implemented?
        h_conv2 = tf.nn.relu(conv2d(h_pool1,w2))

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        flatVecSize = 7*7*64
        #h_pool2_flat = tf.reshape(h_pool2,[-1,flatVecSize,1,1])
        h_pool2_flat = tf.reshape(h_pool2,[-1,flatVecSize])
        w3 = weight_variable([flatVecSize,1024])
        b3 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w3) + b3)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        #h_fc1_flat = tf.reshape(h_fc1,[-1,1024,1,1])
        h_fc1_flat = tf.reshape(h_fc1,[-1,1024])
        w4 = weight_variable([1024,10])
        b4 = bias_variable([10])
        
    # Calculate the final probabilities (logits) for each class
    with tf.name_scope('outputClassProbs'):    
        y_conv = tf.add(tf.matmul(h_fc1_flat,w4), b4, name="classProbs")
    
    return y_conv

# end deepnn

'''
Complete the Graph[10 pts]

We start building the computation graph by creating nodes for the input images 
and target output classes.
'''

# Run with defaults if at highest level
if __name__ == "__main__":  
    
    # Clear checkpoint files to get a clean training run each time
    checkpointSaveDir = "./mnist_cnn_save" # checkpoints saved here
    if os.path.exists(checkpointSaveDir): # only rm if it exists
        shutil.rmtree(checkpointSaveDir, ignore_errors=True)    
    
    # Get the NMIST handwritten-digit data
    x_train, y_train, x_test, y_test = getNMISTData()

    # Placeholders for the data and associated truth
    x = tf.placeholder(tf.float32, [None, 28,28], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    
    # Build the graph for the deep net
    # It is best to literally thing of this as just building the graph, since
    # it is in reality just a "placeholder" or template for what we will
    # actually be running. y_conv is [-1,10], where -1 is the number of input
    # datapoints and 10 is the probability (logit) for each output class 
    # (numeral).
    y_conv = deepnn(x)
    
    #define the loss function (externally from the actual neural net)
    # This is literally just defining the loss function, not actually running it.
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
        
    # Accuracy calculation
    with tf.name_scope('accuracy'):
        correct_prediction = tf.argmax(y_,1)
        y = tf.nn.softmax(y_conv)
        our_prediction = tf.argmax(y,1)
        performance_vec = tf.equal(correct_prediction,our_prediction)
        accuracy = tf.reduce_mean(tf.cast(performance_vec,tf.float32))
        
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
        nEpochs = 2000
        for epoch in range(nEpochs): 
            # Extract data for this batch
            batch = extractBatch(100, x_train, y_train, epoch)
            
            # Run a single epoch with the extracted data batch
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})  
            
            if epoch % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
            
            # Print elapsed time every 100 epochs
            if epoch % 100 == 0:
                curr_sec = time.clock()
                print('    Elapsed time for %d epochs: %g sec' 
                      % (epoch, curr_sec-start_sec))
    
            # Save the model weights (and everything else) every 500 epochs
            if epoch % 500 == 0 or epoch == (nEpochs-1):
                save_path = saver.save(sess, checkpointSaveDir + "/model_at" + str(epoch) + ".ckpt")
                print("    Checkpoint saved to: %s" % save_path)
        

        # Total elapsed time
        end_sec = time.clock()
        print('Total elapsed time for %d epochs: %g sec' 
              % (nEpochs, end_sec-start_sec))
    
        # Finish off by running the test set.  Extract the entire test set.
        test_batch = extractBatch(len(x_test), x_test, y_test, 0)
        print('test accuracy %g' % accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1]}))
        
        # Run a single testpoint through with use_cnn.py as a sanity check
        use_cnn.twoTest()
        
        # Print the location of the saved network
        print("Final trained network saved to: " + save_path)
        print("You can use use_cnn.py with this final network to classify new datapoints")
        