# -*- coding: utf-8 -*-
"""
videoUtilities.py 
DESCRIPTION:
    Functions wrapping cv2 for working with videos
INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

"""
FUNCTION:
    pull_video() 
    
DESCRIPTION:
    Uses cv2 to pull a video frame-by-frame from a video file.
    
INPUTS: 
    inFile: movie file readable by cv2
    
RETURNS: 
    Video from inFile as a [nframes,nx,ny,nColors] 4D array

"""
def pull_video(inFile):
    
    # Open input video stream
    inStream = cv2.VideoCapture(inFile)
    assert inStream.isOpened()
    width,height = int(inStream.get(3)), int(inStream.get(4))
    
    outVideo = np.zeros([0,width,height,3])
       
    try:
        # Read in the video frame-by-frame
        success = True
        iFrame = 0
        while success:
            # Read a frame
            success, frame = inStream.read()
                
            if not success:
                break
           
            outVideo = np.concatenate([outVideo,frame],axis=0)
            
            iFrame += 1
        # while success
        
    finally:       
        # Save it off so we can play the video
        print("Exiting cleanly")
    
    return outVideo
    
# end pull_video()

"""
FUNCTION:
    pull_sequence() 
    
DESCRIPTION:
    Uses cv2 to pull a video frame-by-frame from a series of jpg files.
    
INPUTS: 
    inFileBase: path+prefix to the desired image sequence. For example, to read
        all frames named "frame_####.jpg", set inFileBase="frame_"
OPTIONAL INPUTS:
    ext: filename extension, including '.' (default '.jpg')
    iStart: first frame to read (default 0)
    iEnd: last frame to read (default -1, read until all gone)
    color: set to True to read in color (default False)
    invert: set to True to invert black and white (default False)
    
RETURNS: 
    Video from inFile as a [nframes,nx,ny,nColors] 4D array
"""
def pull_sequence(inFileBase, ext='.jpg', iStart=0, iEnd=-1, color=False, 
                  invert=False):
    index = iStart
    
    # Read first frame to get sizes
    indexStr = "%04d" % index
    fullFilename = inFileBase + indexStr + ext
    if os.path.isfile(fullFilename):
        frame = cv2.imread(fullFilename)
        if not color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #frame = frame[:,:,0] # extract red channel
            if invert:
                frame = cv2.bitwise_not(frame) # fix inverted colors
            nColors = 1
        else:
            nColors = frame.shape[3]
        width, height = frame.shape[:2]
        frame = np.reshape(frame,[1,width,height,nColors]) 
    else:
        print("Problem opening " + fullFilename)
        return -1
        
    #print("Video dimensions: (%d,%d)" % (width, height))
    
    # Initialize output video array
    outVideo = np.zeros([1,width,height,nColors])
    outVideo[0,:,:,:] = frame
    index += 1
    #print("Outvideo dimensions:", outVideo.shape)
    
    while True:
        if index >= iEnd and iEnd != -1:
            break
            
        indexStr = "%04d" % index
        fullFilename = inFileBase + indexStr + ext
        if os.path.isfile(fullFilename):
            frame = cv2.imread(fullFilename) # read the frame
            if not color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                #frame = frame[:,:,0] # extract red channel
                if invert:
                    frame = cv2.bitwise_not(frame) # fix inverted colors
            frame = np.reshape(frame,[1,width,height,nColors]) 
            #print("Frame dimensions: (%d,%d,%d,%d)" % frame.shape)
            outVideo = np.concatenate([outVideo,frame],axis=0) # tack on
        else:
            break # sequence is over
            
        if index % 10 == 9:
            print("Read frame %d" % (index+1))
                        
        index += 1
    
    return outVideo
    
"""
FUNCTION:
    pull_aug_sequence() 
    
DESCRIPTION:
    Uses cv2 to pull an augmented series of images. These images have a file
    format of "[path]/[name]_####_@@@@.[ext]", where "####" represents the 
    temporal frame number, and "@@@@" represents the index of an applied
    augmentation to that frame.  The corresponding truth masks are read in 
    simultaneously to ensure that they match one another.
    
INPUTS: 
    inImageBase: path+prefix to the desired image sequence. For example, to read
        all frames named "augImage_####_@@@@.jpg", set inFileBase="augImage_"
    inMaskBase: path+prefix to the desired mask sequence. For example, to read
        all masks named "augMask_####_@@@@.jpg", set inFileBase="augMask_"
OPTIONAL INPUTS:
    ext: filename extension, including '.' (default '.jpg')
    color: set to True to use color channel (default False)
    
RETURNS: 
    image_stack: Series of augmented frames as a [nframes,nx,ny,nColors] 4D array
    mask_stack:  Series of augmented masks as a [nframes,nx,ny] 3D array
    index_stack: List of strings with format "####_%%%%", where #### is the 
        index of the parent image, and %%%% is the augmentation index
"""
def pull_aug_sequence(inImageBase, inMaskBase, ext='.jpg', color=False):
    imagePath,imagePrefix = os.path.split(inImageBase)
    maskPath, maskPrefix  = os.path.split(inMaskBase)
    
    stackStarted = False
    stackCount = 0
    
    # Pre-read each image filename and create a dict with it
    imageDict = {}
    for imageName in os.listdir(imagePath):
        # Pull only if the image name prefix is in the filename
        if imagePrefix in imageName:
            indexString = imageName[len(imagePrefix):].split('.')[0]
            imageDict.update({indexString: imageName})
    
    # Pre-read each mask filename and create a dict with it
    maskDict = {}
    for maskName in os.listdir(maskPath):
        # Pull only if the mask name prefix is in the filename
        if maskPrefix in maskName:
            indexString = maskName[len(maskPrefix):].split('.')[0]
            maskDict.update({indexString: maskName})
    
    
    # Loop over each image, find matching mask if it is there, and read both
    for indexString, imageName in imageDict.items():
        if indexString in maskDict: # if matching mask exists
            if stackCount % 100 == 99:
                print("Read %d images" % (stackCount+1))
            
            # Read in both the image and the mask as a pair
            image = cv2.imread(inImageBase+indexString+ext)
            mask  = cv2.imread(inMaskBase +indexString+ext)
            width, height, nColors = image.shape
            if not color:
                image = image[:,:,0] # just pull 1st channel
                nColors = 1
            mask = mask[:,:,0] # just pull 1st channel, masks are always B/W
            
            # Preallocate image and mask stacks for speed
            if not stackStarted:
                stackStarted = True
                imageStack = np.zeros([len(imageDict),width,height,nColors])
                maskStack  = np.zeros([len(imageDict),width,height]) == 1
                indexStack = ['' for i in range(len(imageDict))]
                        
            # Reshape nicely and add to the stacks
            image = np.reshape(image,[1,width,height,nColors])
            mask  = np.reshape(mask, [1,width,height]) < 1
            imageStack[stackCount,:,:,:] = image
            maskStack[stackCount,:,:]  = mask
            indexStack[stackCount] = indexString
            stackCount += 1
        else:
            print("WARNING: Could not find a mask matching " + imageName+indexString+ext)
    # for all images
    
    # Trim off any unmatched images and masks
    imageStack = imageStack[:stackCount,:,:,:]
    maskStack  = maskStack [:stackCount,:,:]
    indexStack = indexStack[:stackCount]
    
    # Decolorize if so desired
    if not color:
        imageStack = imageStack[:,:,:,0]
        
    # Normalize image
    imageStack /= 255.0
    
    return imageStack, maskStack, indexStack
    
"""
FUNCTION:
    augment_sequence() 
    
DESCRIPTION:
    Uses cv2 to pull a video and the corresponding truth mask frame-by-frame 
    from a series of jpg files, then applies random scaling and cropping to
    extract a region of a given shape matching the desired NN input shape.
    This CAN and SHOULD be able to completely remove the target from the image.
    We want to have examples with no target present in our training sets!
    Finally a series of random noise, brightness, contrast, and rotation 
    augmentations are performed. This produces a bevy of augmented images for 
    each true input image.  Results are saved to file. Every output image has
    the same dimensions, as set by the outShape parameter.
    
INPUTS: 
    inImageFileBase: path+prefix to the desired image sequence. For example, to 
        read all frames named "frame_####.jpg", set inImageFileBase="frame_"
    inMaskFileBase: path+prefix to the desired mask sequence that goes with 
        inImageFileBase. For example, to read all masks named "mask_####.jpg",
        set inMaskFileBase="mask_".
    outputFolder: path to the folder where the augmented images and masks will
        be saved to.
OPTIONAL INPUTS:
    ext: filename extension, including '.' (default '.jpg')
    iStart: first frame to read (default 0)
    iEnd: last frame to read (default -1, read until all gone)
    color: set to True to read in color (default False)
    invert: set to True to invert black and white (default False)
    outShape: forced [width,height] of the augmented images (default [256,256])
    
OUTPUTS:
    Writes augmented versions of each input image and corresponding mask to
    the outputFolder with the name "augmented_####_a****.jpg", where #### 
    inherits from the frame number of the input parent image, and **** uniquely
    identifies the augmentation applied.
RETURNS: 
    None
"""
def augment_sequence(inImageFileBase, inMaskFileBase, outputFolder,
                     ext='.jpg', iStart=0, iEnd=-1, color=False, 
                     invert=False, outShape=[256,256]):
    
    print("Applying random augmentations to images in " + inImageFileBase)
    
    # Make output directory if it does not yet exist
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    
    # Read in the images to augment one-by-one to preserve memory
    rawIndex = iStart
    while True:
        if rawIndex >= iEnd and iEnd != -1:
            break
        
        # Use pull_sequence with iStart=iEnd to just get one at a time
        print("Pulling image %d from file" % rawIndex)
        rawImage = pull_sequence(inImageFileBase, iStart=rawIndex, iEnd=rawIndex,
                                 ext=ext, color=color, invert=invert)
        print("  Pulling corresponding mask sequence from file")
        rawMask = pull_sequence(inMaskFileBase, iStart=rawIndex, iEnd=rawIndex,
                                 ext=ext, color=False, invert=False)
        rawImage = np.squeeze(rawImage) # remove singleton dimensions
        rawMask = np.squeeze(rawMask) # remove singleton dimensions
        if len(rawImage.shape) == 1:
            if rawImage == -1 or rawMask == -1:
                break # we reached the end or had problems reading
            
        iAugmentation = 0
        
        # Apply some random crops and resizes to the raw images and masks, 
        # always applying the same crop and resize to the image and its
        # corresponding mask.
        # Determine valid resizes and draw some randomly
        nbRandomCropResize = 8   
        nbRandomBrightContrast = 4
        nbRandomNoise = 4
        print("Applying %d augmentations" % (nbRandomCropResize*nbRandomBrightContrast))
        for iCropResize in range(nbRandomCropResize):
            #print("  Applying random crop/resize %d of %d" % (iCropResize,nbRandomCropResize))
            # Resize the image and mask
            resizedImage, resizedMask = random_resize(rawImage, rawMask, outShape)
            
            # Rotate here?? Ensure only valid area is cropped to
            #rotatedImage, rotatedMask = random_rotation(resizedImage, resizedMask, outShape)
                        
            # Crop the image and mask randomly
            croppedImage, croppedMask = random_crop(resizedImage, resizedMask, outShape)
                        
            # Apply random brightness/contrast adjustments to the image
            for iBC in range(nbRandomBrightContrast):
                bcImage = random_bright_contrast(croppedImage)
                bcMask = croppedMask # brightness/contrast do not effect the mask
            
                # Set as final image to output
                finalAugmentedImage = bcImage
                finalAugmentedMask = bcMask
            
                # Write the augmented image and mask to file
                augImageFileStr = os.path.join(outputFolder,
                    'augImage_%04d_%04d' % (rawIndex, iAugmentation) + ext)
                augMaskFileStr = os.path.join(outputFolder,
                    'augMask_%04d_%04d' % (rawIndex, iAugmentation) + ext)
                cv2.imwrite(augImageFileStr,np.squeeze(finalAugmentedImage))
                cv2.imwrite(augMaskFileStr, np.squeeze(finalAugmentedMask))
                iAugmentation += 1
        
        rawIndex += 1
    # end while loop over raw images
    
    print("Augmented images written to " + outputFolder)
    
# end augment_sequence()
  
'''
FUNCTION:
    random_resize() 
    
DESCRIPTION:
    Applies a random valid resize to the input image and corresponding mask.
    Always leaves enough pixels to crop to the designated output shape.
    
INPUTS: 
    image: image to resize (2D for now, no colors)
    mask: mask to resize. Must be same shape as image
    outShape: Output shape that image and mask will be cropped to
RETURNS: 
    resizedImage: image resized randomly
    resizedMask:  mask resized randomly to match image
'''
def random_resize(image, mask, outShape):
    # Determine smallest scale factor that will still allow an outShape image
    # to be extracted afterwards.
    lowScl = np.max(np.divide(outShape,image.shape[1:3])) 
    highScl = 2*lowScl
    scaleFactor = np.random.uniform(low=lowScl, high=highScl)
    outSizeSet = [np.ceil(dim * scaleFactor).astype(int) for dim in image.shape[1::-1]]
    
    # Resize the image and mask
    resizedImage = cv2.resize(image, tuple(outSizeSet))
    resizedMask  = cv2.resize(mask, tuple(outSizeSet))
    
    return resizedImage, resizedMask
  
'''
FUNCTION:
    random_crop() 
    
DESCRIPTION:
    Applies a random valid crop to the input image and corresponding mask.
    Always crops to the designated output shape
    
INPUTS: 
    image: image to crop (2D for now, no colors)
    mask: mask to crop. Must be same shape as image
    outShape: Output shape to crop image and mask to
RETURNS: 
    croppedImage: random outShape block extracted from image
    croppedMask:  random outShape mask  extracted from image
'''
def random_crop(image, mask, outShape):
    # Given the resizing, determine some valid crop areas
    maxValidCropStart = [image.shape[i] - outShape[i] for i in range(2)]
    cropSet = [np.floor(np.random.uniform(low=0,high=maxValidCropStart[0])).astype(int), \
               np.floor(np.random.uniform(low=0,high=maxValidCropStart[1])).astype(int)]
    
    # Apply the cropping
    croppedImage = image[cropSet[0]:cropSet[0]+outShape[0], \
                         cropSet[1]:cropSet[1]+outShape[1]]
    croppedMask  = mask [cropSet[0]:cropSet[0]+outShape[0], \
                         cropSet[1]:cropSet[1]+outShape[1]]
                         
    return croppedImage, croppedMask  
    
  
'''
FUNCTION:
    random_rotation() 
    
DESCRIPTION:
    Applies a random valid rotation to the input image and corresponding mask.
    
INPUTS: 
    image: image to rotate (2D for now, no colors)
    mask: mask to rotate. Must be same shape as image
    outShape: Output shape that image and mask will be cropped to
OPTIONAL INPUTS:
    rotationBoundsDeg: [smallest,largest] allowed rotation in degrees CCW 
                       (default [-30,+30])
RETURNS: 
    rotImage: image with random rotation applied
    rotMask: mask with randome rotation applied
'''
def random_rotation(image, mask, outShape, rotationBoundsDeg=[-30.0,30.0]):
    # Determine a random rotation angle
    angle = np.random.uniform(low=rotationBoundsDeg[0],high=rotationBoundsDeg[1])
    
    # Rotate the image and mask by the random angle
    rotImage = cv2.imutils.rotate(image,angle)
    rotMask = cv2.imutils.rotate(mask,angle)
                         
    return rotImage, rotMask  


'''
FUNCTION:
    random_bright_contrast() 
    
DESCRIPTION:
    Applies a random brightness and contrast adjustment to the input image
    
INPUTS: 
    image: image to adjust (2D for now, no colors)
OPTIONAL INPUTS:
    alphaRange: allowable range of contrast adjustments (default [0.7,1.3])
    betaRange: allowable range of brightness adjustments (default [0.7,1.3])
RETURNS: 
    bcImage: image with adjusted contrast and brightness
'''
def random_bright_contrast(image,alphaRange=[0.7,1.3],betaRange=[0.7,1.3]):
    # Determine a random alpha (contrast adjustment)
    alpha = np.random.uniform(low=alphaRange[0],high=alphaRange[1])
    
    # Determine a random beta (brightness adjustment)
    beta  = np.random.uniform(low=betaRange[0], high=betaRange[1])
    
    # Apply the contrast adjustment (alpha), and brightness adjustment (beta)
    bcImage = cv2.add(cv2.multiply(image,alpha), beta)
    
    return bcImage

'''
FUNCTION:
    train_test_split() 
    
DESCRIPTION:
    Applies a random train/test split to the images and masks with a given 
    fraction as training data.
    
INPUTS: 
    images: stack of images [nBatch,width,height,nChannels]
    masks: stack of corresponding masks [nBatch,width,height]
OPTIONAL INPUTS:
    trainFraction: fraction of images as training (default 0.8)
RETURNS: 
    train_images: trainFraction of the input images
    train_masks: trainFraction of the input masks corresponding to train_images
    test_images: the other (1-trainFraction) of the input images
    test_masks: the other (1-trainFraction) of the input masks corresponding to test_images
'''
def train_test_split(images, masks, trainFraction=0.8):
    nBatch = images.shape[0]
    indices = [i for i in range(nBatch)]
    random.shuffle(indices)
    nbTrain = int(nBatch*trainFraction)
    train_images = images[indices[:nbTrain],:,:]
    test_images = images[indices[nbTrain:],:,:]
    train_masks = masks[indices[:nbTrain],:,:]
    test_masks =  masks[indices[nbTrain:],:,:]
    
    return train_images, train_masks, test_images, test_masks
    

'''
FUNCTION:
    train_test_split_noCheat() 
    
DESCRIPTION:
    Applies a random train/test split to the images and masks with a given 
    fraction as training data. Takes into account the fact that images are
    frequently augmented duplicates of one another and will not put augmented
    duplicates in both train and test sets.
    
INPUTS: 
    images: stack of images [nBatch,width,height,nChannels]
    masks: stack of corresponding masks [nBatch,width,height]
    indices: list of corresponding indices with the format ####_@@@@, where
        #### represents the parent image index, and @@@@ represents the
        augmentation index.
OPTIONAL INPUTS:
    trainFraction: fraction of images as training (default 0.8)
RETURNS: 
    train_images: trainFraction of the input images
    train_masks: trainFraction of the input masks corresponding to train_images
    test_images: the other (1-trainFraction) of the input images
    test_masks: the other (1-trainFraction) of the input masks corresponding to test_images
'''
def train_test_split_noCheat(images, masks, indices, trainFraction=0.8):
    # Generate a dictonary of the parent indices
    parentList = {}
    for index,indexString in enumerate(indices):
        parentID = indexString[:4]
        if not parentID in parentList:
            parentList[parentID] = [] # initialize
        parentList[parentID].append(index) # add this index to the child list
    # endfor    
    
    # Divvy up the parent indices into train and test batches
    nParents = len(parentList)
    parentIndices = [i for i in range(nParents)]
    random.shuffle(parentIndices)
    nbTrain = int(nParents*trainFraction)
    trainParentIndices = parentIndices[:nbTrain] # just pull the first nbTrain of the shuffled indices
    
    # Assign all the child images from each parent image to test or train per 
    # the above assignment.
    i = 0
    trainIDList = []
    testIDList = []
    # Loop over each parent ID
    for parentID, children in parentList.items():
        for childID in children:
            if i in trainParentIndices:
                # Put this image into the training set
                trainIDList.append(childID)
            else:
                # Put this image into the test set
                testIDList.append(childID)
        i += 1 # go to next parentID
    # endfor parent IDs
    
    # Put each image and mask into the appropriate set
    train_images = images[trainIDList,:,:]
    test_images = images[testIDList,:,:]
    train_masks = masks[trainIDList,:,:]
    test_masks =  masks[testIDList,:,:]
    
    return train_images, train_masks, test_images, test_masks


# Testing here
if __name__ == "__main__":
    '''
    myVideo = pull_sequence(os.path.join("sequences",
        "defaultGreenscreenVideo_over_PHO_hallway","frame_"), 
        iStart=330, iEnd=360, invert = True)
    print("nFrames: %d" % myVideo.shape[0])
    
    plt.imshow(np.squeeze(myVideo[0,:,:,:]),cmap='Greys')
    plt.show()
    '''
    '''
    myMaskVideo = pull_sequence(os.path.join("sequences",
        "defaultGreenscreenVideo_over_PHO_hallway","mask_"), iStart=330, iEnd=360)
    print("nFrames: %d" % myVideo.shape[0])
    
    plt.imshow(np.squeeze(myMaskVideo[0,:,:,:]),cmap='Greys')
    plt.show()
    '''
    
    if not os.path.exists("augmentedSequences"):
        os.mkdir("augmentedSequences")
    '''
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_PHO_hallway","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_PHO_hallway","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_PHO_hallway_64x64"),
                     iStart=250, iEnd=-1, outShape=[64,64])
    '''
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x64"),
                     iStart=175, iEnd=637, outShape=[64,64])
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab2_64x64"),
                     iStart=175, iEnd=637, outShape=[64,64])
    
