import tempfile

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
        x_image = tf.manip.reshape(x,[N,28,28,1])

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
    
# For saving the graph, DO NOT CHANGE.
graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

'''
Train and Evaluate the Model[5 pts]

<<<<<<< HEAD
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
=======
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

tf.logging.set_verbosity(tf.logging.INFO)

# This function defines the forward pass of the network. The backward pass is
# executed implicitly by tensorflow.
# The default implementation has 2 hidden conv+maxpool layers followed by a 
# single fully-connected (dense) layer.
# INPUTS:
#     features: The data itself.  [nImages,nx,ny,nColors]. Dictonary for some reason
#     labels: True classes of each image in features
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input layer.
    # Note that features["x"] just extracts that part of the dictionary
    # The -1 indicates a "wildcard" dimension, in this case the number of
    # images to use in the training.
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # 32 5x5 kernels, maintain resolution, relu
    conv1 = tf.keras.layers.Convolution2D(
        filters=32,
        kernel_size=[5, 5], # original
        padding="same",
        activation=tf.nn.relu)(input_layer)
        
    # Pooling Layer #1
    # 2x2 pool on 28x28 gives makes pool1 size 14x14
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)

    # Convolutional Layer #2
    # 64 5x5 kernels acting on 14x14 images, maintain resolution, relu
    conv2 = tf.keras.layers.Convolution2D(
        filters=64,
        kernel_size=[5, 5], # original
        padding="same",
        activation=tf.nn.relu)(pool1)
        
    # Pooling Layer #2
    # 2x2 pool on 14x14 gives pool2 size 7x7
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)

    # Dense (fully connected) Layer
    # Resize 7x7x64 per input pool2 to be a single vector per input
    # dense and then dropout here are still large
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(pool2_flat)
    dropout = tf.keras.layers.Dropout(
        rate=0.4)(dense,mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer (final classes)
    # Final layer to get down to 10 classes per input
    logits = tf.keras.layers.Dense(units=10)(dropout)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # If we are simply running a forward pass for prediction (actual runtime
    # usage), then just return that.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        #@@@"accuracy": tf.keras.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
      
      
      
# Run with defaults if at highest level
if __name__ == "__main__":  
    # Clear checkpoint files to get a clean training run each time
    checkpointSaveDir = "./mnist_convnet_model" # checkpoints saved here
    if os.path.exists(checkpointSaveDir): # only rm if it exists
        shutil.rmtree(checkpointSaveDir)    
    
    # Load training and eval data
    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    print("Number of training images: " + str(train_data.shape[0]))
    print("Number of testing  images: " + str(eval_data.shape[0]))
    print("Image shape (nx,ny): (" + str(train_data.shape[1]) + 
                               "," + str(train_data.shape[2]) + ")")

    train_data = train_data/np.float32(255) # normalize images
    train_labels = train_labels.astype(np.int32)  # not required

    eval_data = eval_data/np.float32(255) # normalize images
    eval_labels = eval_labels.astype(np.int32)  # not required
    
    # Display some example data
    chain = np.squeeze(train_data[0,:,:])
    for i in range(15):
        chain = np.append(chain,np.squeeze(train_data[i+1,:,:]),axis=1)
    print("Example training data")
    plt.figure()
    plt.imshow(chain)
    labelstrs = "".join([str(truth) + ", " for truth in train_labels[:16]])
    print("Truth: ", labelstrs)
    
    
    # Create the classifier object (estimator)
    # Note that THIS IS WHERE CNN_MODEL_FN GETS REFERENCED!!!
    print("Initializing estimator")
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=checkpointSaveDir)
        
       
    # Set up logging for predictions
    #tensors_to_log = {"probabilities": "softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(
    #    tensors=tensors_to_log, every_n_iter=50)
        
    # Configure the training function
    print("Configuring training function")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, # x is the training data
        y=train_labels, # y is the testing data
        batch_size=100, # number of images per batch (within an epoch)
        num_epochs=None,
        shuffle=True)

    # Train just one step and display the probabilties for sanity check
    #mnist_classifier.train(input_fn=train_input_fn,steps=1,hooks=[logging_hook])
    
    # Train for another 1000 epochs (do the grunt work)
    print("Executing training")
    # This is where all the deprecation warnings come in
    mnist_classifier.train(input_fn=train_input_fn, steps=3000)
    
    # Configure the evaluation
    print("Configuring evaluation")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    # Run the evaluator and print the results
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
    checkpoint_file=tf.train.latest_checkpoint(checkpointSaveDir)
    print("Trained model saved to: " + checkpoint_file)
>>>>>>> 53cf29edc5ab17bc9244ab49c00b95f4d3a44e6e

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
