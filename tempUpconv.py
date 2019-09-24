import tensorflow as tf
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
    N, nx, ny = x.shape[:3]
    nkx, nky, nW = W.shape[:3] # nwIn, nwOut here instead???
    
    # Output dimension and stride calculations
    outShape = tf.stack([-1, nkx*(nx-1)+1, nky*(ny-1)+1, nW])
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
    return tf.concat([xNew, xOld], 0)


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
#ADD THIS TO run_cnn.py
"""
# Add to main level near the top
    y_train_pmMask = booleanMaskToPlusMinus(makeThresholdMask(x_train))
    y_test_pmMask  = booleanMaskToPlusMinus(makeThresholdMask(x_test))
    
    y_pmMask = tf.placeholder(tf.float32, [None, 28,28], name="y_mask")
    

# Add to deepnn
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
        h_sk1 = addSkipConnection(h_upconv1, h_pool1) # skip connection
        wu1 = weight_variable([2,2,1,32])
        heatmap = tf.nn.relu(upconv2d(h_sk1, wu1)) # [-1,28,28,1]
        
    # The size of heatmap here should be [batch,28,28,1] for NMIST
    
    
"""
Gain calculation followed by optimization. This will replace the loss,
optimization, and accuracy name scopes of the original NMIST version.
"""    
    # The heatmap loss calculation
    with tf.name_scope('heatmapGain'):
        # Gain here is really just an pixel-wise heatmap*truthmask product
        # + gain for every heatmap pixel that IS     part of the targetmask
        # - gain for every heatmap pixel that IS NOT part of the targetmask
        # To do this, targetmask must have +1's at target     locations
        # and         targetmask must have -1's at background locations
        # Make sure targetmask is formed in this way!!!
        gainmap = tf.multiply(heatmap, y_pmMask) # pixel-by-pixel gain
        
        # May be useful to have an intermediate reduction here of a single
        # gain value for each individual image...
        
        # Average of gain across every pixel of every image
        gain = tf.reduce_mean(tf.cast(gainmap,tf.float32)) 
        
    # Optimization calculation
    with tf.name_scope('adam_optimizer'):
        # Basic ADAM optimizer, maximizing gain rather than minimizing loss
        train_step = tf.train.AdamOptimizer().maximize(gain)
        
    # Final accuracy calculation is nothing more than the gain itself. No need
    # for an additional calculation.