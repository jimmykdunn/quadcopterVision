# -*- coding: utf-8 -*-
"""
FUNCTION: run_cnn
DESCRIPTION:
    Trains and evaluates a convolutional neural net (CNN) to generate a heatmap
    of the probability of a quadcopter being at each pixel in the input image.
    Uses tensorflow.  Based on NMIST assignment in CS540 Machine Learning 
    course at Boston University in November 2018.
    
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
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# IMPLEMENT THIS
# NOTE: FOR ALL THE FOLLOWING CODES, DO NOT IMPLEMENT YOUR OWN VERSION. 
# USE THE BUILT-IN METHODS FROM TENSORFLOW.
# Take a look at TensorFlow API Docs.
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  # Input x is 4d [image_num, x, y, chan] (chan = color) [?,28,28,1]
  # Input W (kernel) is also 4d [height, width, in chan, out chan], [28,28,1,1]
  stride = [1,1,1,1] # stride of filter.  Full stride means 1
  padding = 'SAME' # full zeropadding
  return tf.nn.conv2d(x,W,stride,padding)

# IMPLEMENT THIS
def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  # Input x is 4d [image_num, x, y, chan] (chan = color) [?,28,28,1]
  ksize = [2,2,1,1] # size of pooling (each dimension of the kernel)
  strides = [1,1,1,1] # stride size
  padding = 'SAME'
  return tf.nn.max_pool(x,ksize,strides,padding)

'''
Build the CNN

First Convolutional Layer[10 pts]
We can now implement our first layer. It will consist of convolution, followed 
by max pooling. The convolution will compute 32 features for each 5x5 patch. 
Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions
 are the patch size, the next is the number of input channels, and the last is
 the number of output channels. We will also have a bias vector with a 
 component for each output channel.

To apply the layer, we first reshape x to a 4d tensor, with the second and 
third dimensions corresponding to image width and height, and the final 
dimension corresponding to the number of color channels.

We then convolve x_image with the weight tensor, add the bias, apply the ReLU 
function, and finally max pool. The max_pool_2x2 method will reduce the image 
size to 14x14.

Second Convolutional Layer[5 pts]
In order to build a deep network, we stack several layers of this type. The
 second layer will have 64 features for each 5x5 patch.
 
Fully Connected Layer[10 pts]
Now that the image size has been reduced to 7x7, we add a fully-connected 
layer with 1024 neurons to allow processing on the entire image. We reshape 
the tensor from the pooling layer into a batch of vectors, multiply by a 
weight matrix, add a bias, and apply a ReLU.

SoftmaxLayer[5 pts]
Finally, we add a layer of softmax regression
'''

def deepnn(x):
    """
    deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    N = x.shape[0]  # number of images in the dataset (N_examples)
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x,[N,28,28,1])

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
        h_pool2_flat = tf.manip.reshape(h_pool2,[N,flatVecSize,1,1])
        w3 = weight_variable([flatVecSize,1024])
        b3 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w3) + b3)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        h_fc1_flat = tf.manip.reshape(h_fc1,[N,1024,1,1])
        w4 = weight_variable([1024,10])
        b4 = bias_variable([10])
        y_conv = tf.nn.relu(tf.matmul(h_fc1_flat,w4) + b4)
    return y_conv

'''
Complete the Graph[10 pts]

We start building the computation graph by creating nodes for the input images 
and target output classes.
'''
# Import data
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Build the graph for the deep net
y_conv = deepnn(x)

'''
We can specify a loss function just as easily. Loss indicates how bad the 
model's prediction was on a single example; we try to minimize that while 
training across all the examples. Here, our loss function is the cross-entropy 
between the target and the softmax activation function applied to the model's 
prediction. As in the beginners tutorial, we use the stable formulation:
'''

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    
'''
First we'll figure out where we predicted the correct label. tf.argmax is an 
extremely useful function which gives you the index of the highest entry in a 
tensor along some axis. For example, tf.argmax(y,1) is the label our model 
thinks is most likely for each input, while tf.argmax(y_,1) is the true label. 
We can use tf.equal to check if our prediction matches the truth.

That gives us a list of booleans. To determine what fraction are correct, we 
cast to floating point numbers and then take the mean. For example, [True, 
False, True, True] would become [1,0,1,1] which would become 0.75.
'''
with tf.name_scope('accuracy'):
    correct_prediction = tf.argmax(y_,1)
    our_prediction = tf.argmax(y_conv,1)
    performance_vec = tf.equal(correct_prediction,our_prediction)
    accuracy = tf.reduce_mean(float(performance_vec))
    
graph_location = 'tfGraph.gph'
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

'''
Train and Evaluate the Model[5 pts]

We will use a more sophisticated ADAM optimizer instead of a Gradient Descent 
Optimizer.

We will add logging to every 100th iteration in the training process.

Feel free to run this code. Be aware that it does 20,000 training iterations 
and may take a while (possibly up to half an hour), depending on your processor.

The final test set accuracy after running this code should be approximately 
99.2%.

We have learned how to quickly and easily build, train, and evaluate a fairly 
sophisticated deep learning model using TensorFlow.

'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
