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
#import matplotlib.pyplot as plt
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
    imageStack: Series of augmented frames as a [nframes,nx,ny,nColors] 4D array
    maskStack:  Series of augmented masks as a [nframes,nx,ny] 3D array
    indexStack: List of strings with format "####_%%%%", where #### is the 
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
    sort_aug_sequence() 
    
DESCRIPTION:
    Takes in an unordered but matching list of images, masks, and index strings,
    then sorts them first by augmentation index (last 4 characters in index
    string) and second by temporal index (first 4 characters in index string).
    
INPUTS: 
    imageStack: Series of augmented frames as a [nframes,nx,ny] 3D array
    maskStack:  Series of augmented masks as a [nframes,nx,ny] 3D array
    indexStack: List of strings with format "####_%%%%", where #### is the 
        index of the parent image, and %%%% is the augmentation index
    
RETURNS: 
    image_stack_sorted: 4D array of images [nAugs,nframes,nx,ny], sorted
    mask_stack_sorted: 4D array of masks [nAugs,nframes,nx,ny], sorted
    index_stack_sorted: 2D list of strings [nAugs][nframes], sorted
"""
def sort_aug_sequence(imageStack, maskStack, indexStack):
    print("Sorting augmented image sequences")
    
    # Dimensions of arrays
    nImages,width,height = imageStack.shape[:3]

    # First, we gather together each unique augmentation index
    allAugIndices = [int(strIndex[-4:]) for strIndex in indexStack]
    allTemporalIndices = [int(strIndex[:4]) for strIndex in indexStack]
    augIndices = np.unique(allAugIndices)
    temporalIndices = np.unique(allTemporalIndices)
    
    # Sort the augmented and temporal indices
    augIndices.sort()
    temporalIndices.sort()
    nAugs = len(augIndices)
    nFrames = len(temporalIndices)
    
    # Allocate sorted lists ahead of time for speed
    image_stack_sorted = np.zeros([nAugs,nFrames,width,height])
    mask_stack_sorted  = np.full([nAugs,nFrames,width,height],False)
    index_stack_sorted = [['####_****']*nFrames]*nAugs
    
    # Loop over augmentations first
    # Enumerations used to deal with case when augIndex or temporalIndex do not
    # start at zero.
    for i_aug, augIndex in enumerate(augIndices):
        for i_temp, temporalIndex in enumerate(temporalIndices):
            # Form the full string index out of the temporal and aug indices
            indexStr = "%04d_%04d" % (temporalIndex,augIndex)
            
            # Ensure this combo actually exists
            if indexStr in indexStack:
                i = indexStack.index(indexStr)
                image_stack_sorted[i_aug,i_temp,:,:] = imageStack[i,:,:]
                mask_stack_sorted[i_aug,i_temp,:,:] = maskStack[i,:,:]
                index_stack_sorted[i_aug][i_temp] = indexStack[i]
            else:
                print("WARNING: Missing augmentation % for frame %" % (augIndex,temporalIndex))
                continue
        # end for temporalIndex
    # end for augIndex

    print("Sorting complete")
    return image_stack_sorted, mask_stack_sorted, index_stack_sorted

# end sort_aug_sequence()
    

"""
FUNCTION:
    extract_random_frame() 
    
DESCRIPTION:
    Extracts a single frame from the input stack of images randomly
    
INPUTS: 
    imageStack: Series of augmented frames as a [nframes,nx,ny] 3D array
    maskStack:  Series of augmented masks as a [nframes,nx,ny] 3D array
    indexStack: List of strings with format "####_%%%%", where #### is the 
        index of the parent image, and %%%% is the augmentation index
    
RETURNS: 
    image: randomly selected image from the stack
    mask: corresponding binary mask
    index: corresponding index string
"""
def extract_random_frame(imageStack, maskStack, indexStack):

    nFrames = imageStack.shape[0]

    # Pull a random number and use it to select a single frame
    idx = np.floor(np.random.uniform(low=0, high=nFrames-1)).astype(int)

    return imageStack[idx,:,:], maskStack[idx,:,:], indexStack[idx]
# end extract_random_frame


"""
FUNCTION:
    find_siamese_match() 
    
DESCRIPTION:
    Takes the input index string and returns the image, mask, and index string
    of its "siamese match" - the frame which we expect it to be (nearly)
    identical to from a target heatmap perspective.  By default, this is a 
    neighboring temporal frame with the same augmentation index to ensure
    an indentical crop, scaling, and brightness/contrast adjustment.
    
INPUTS: 
    indexString: String of image to find the siamese match for
    imageStack: Series of augmented frames as a [nframes,nx,ny] 3D array
    maskStack:  Series of augmented masks as a [nframes,nx,ny] 3D array
    indexStack: List of strings with format "####_%%%%", where #### is the 
        index of the parent image, and %%%% is the augmentation index
    
OPTIONAL INPUTS:
    offset: number of frames after the input index to pull as the siamese match.
            This can be negative to pull a frame in the past. Default 1.
    randomSign: Set to true for the sign of offset to be randomly determined.
            This makes temporal order irrelevant - in particular removing bias
            due to a consistent direction of motion.
    
RETURNS: 
    pairedImage: siamese matched index
    pairedMask: siamese matched mask
    pariedIndexString: siamese matched index string. Will be "****_****" if a
        siamese match for the input image is not found. This should be checked
        for and dealt by the calling function.
"""
def find_siamese_match(indexString, imageStack, maskStack, indexStack, 
    indexStack_plus, offset=1, randomSign=False):
    
    # Number of images available
    nImages = imageStack.shape[0]
    
    # Extract this frame's indices
    temporalIndex = int(indexString[:4])
    augIndex = int(indexString[5:9])
    vidIndex = int(indexString[-2:])
    
    # Randomize offset direction if desired
    if randomSign:
        if np.random.rand() < 0.5: # 50% shot of flipping sign
            offset = int(-1*offset)
    
    # Determine temporal index of siamese match
    matchedTemporalIndex = temporalIndex + offset
    matchedAugIndex = augIndex
    matchedVidIndex = vidIndex
    
    matchedIndexStr = "%04d_%04d_%02d" % (matchedTemporalIndex,matchedAugIndex,matchedVidIndex)
    
    if matchedIndexStr in indexStack_plus:
        pairIdx = indexStack_plus.tolist().index(matchedIndexStr) # this may be a speed bottleneck, could use preformed dict to speedup
        pairedImage = imageStack[pairIdx,:,:]
        pairedMask = maskStack[pairIdx,:,:]
        pairedIndexString = indexStack[pairIdx]
    else:
        # No appropriate siamese match found (likely the input frame is too 
        # near to the start or end of the sequence)
        pairedImage = np.zeros(imageStack.shape[1:3])   
        pairedMask = np.zeros(imageStack.shape[1:3]) 
        pairedIndexString = "****_****" # flags match not found
        
        
    return pairedImage, pairedMask, pairedIndexString

# end find_siamese_match
  
    
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
  
  

"""
FUNCTION:
    augment_continuous_sequence() 
    
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
    
    This "continuous" version applies the same brightness and crop/scaling
    to every image in the sequence, so that each augment index can be played
    as a movie and will appear normally without hopping around in space or 
    brightness.
    
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
def augment_continuous_sequence(inImageFileBase, inMaskFileBase, outputFolder,
                     ext='.jpg', iStart=0, iEnd=-1, color=False, 
                     invert=False, outShape=[256,256]):
    
    print("Applying random augmentations to images in " + inImageFileBase)
    
    # Make output directory if it does not yet exist
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    
    # Determine valid resizes and draw some randomly
    nbRandomCropResize = 8   
    nbRandomBrightContrast = 4
    nbMirror = 2
    nbRandomNoise = 4
    print("Applying %d augmentations" % (nbRandomCropResize*nbRandomBrightContrast*nbMirror))
    
    # Loop over crop/resize pairs
    iAugmentation = 0
    for iCropResize in range(nbRandomCropResize):
        # Loop over brightness/contrast adjustments
        for iBC in range(nbRandomBrightContrast):
            # Make a seed for random number generator that we will apply 
            # uniformly across each image in this video.
            randSeed = 7*(iCropResize*nbRandomBrightContrast+iBC)
            
            for iMirror in range(nbMirror):
                print("Executing random crop/resize %d of %d with random brightness/contrast %d of %d, mirror %d of %d" % 
                (iCropResize+1,nbRandomCropResize,iBC+1,nbRandomBrightContrast,iMirror+1,nbMirror))
        
        
                # Read in the images to augment one-by-one to preserve memory
                rawIndex = iStart
                while True:
                    if rawIndex >= iEnd and iEnd != -1:
                        break
                    
                    # Use pull_sequence with iStart=iEnd to just get one at a time
                    rawImage = pull_sequence(inImageFileBase, iStart=rawIndex, iEnd=rawIndex,
                                             ext=ext, color=color, invert=invert)
                    rawMask = pull_sequence(inMaskFileBase, iStart=rawIndex, iEnd=rawIndex,
                                             ext=ext, color=False, invert=False)
                    rawImage = np.squeeze(rawImage) # remove singleton dimensions
                    rawMask = np.squeeze(rawMask) # remove singleton dimensions
                    if len(rawImage.shape) == 1:
                        if rawImage == -1 or rawMask == -1:
                            break # we reached the end or had problems reading
                    
                    # Set the numpy random seed manually so that it applies the
                    # same resize, crop, brightness and contrast settings to every
                    # video in teh sequence.
                    np.random.seed(randSeed)
                                    
                    # Apply the random resize for this video
                    resizedImage, resizedMask = random_resize(rawImage, rawMask, outShape)
                        
                    # Rotate here?? Ensure only valid area is cropped to
                    #rotatedImage, rotatedMask = random_rotation(resizedImage, resizedMask, outShape)
                                    
                    # Apply the appropriate random crop for this video
                    croppedImage, croppedMask = random_crop(resizedImage, resizedMask, outShape)
                                    
                    # Apply the random brightness/contrast adjustments to the image
                    bcImage = random_bright_contrast(croppedImage)
                    bcMask = croppedMask # brightness/contrast do not effect the mask
                        
                    # Apply mirroring if second time thru, else do nothing
                    if iMirror == 1:
                        mirrorImage = np.flip(bcImage,axis=1)
                        mirrorMask = np.flip(bcMask,axis=1)
                    else:
                        mirrorImage = bcImage
                        mirrorMask = bcMask
                        
                    # Set as final image to output
                    finalAugmentedImage = mirrorImage
                    finalAugmentedMask = mirrorMask
                
                    # Write the augmented image and mask to file
                    augImageFileStr = os.path.join(outputFolder,
                        'augImage_%04d_%04d' % (rawIndex, iAugmentation) + ext)
                    augMaskFileStr = os.path.join(outputFolder,
                        'augMask_%04d_%04d' % (rawIndex, iAugmentation) + ext)
                    cv2.imwrite(augImageFileStr,np.squeeze(finalAugmentedImage))
                    cv2.imwrite(augMaskFileStr, np.squeeze(finalAugmentedMask))
                    
                    rawIndex += 1
                # end while loop over raw images
                iAugmentation += 1 # increment augmentation counter
            # end for loop over mirrors
        # end for loop over brightness/contrasts
    # end for loop over crop/resizes
    
    print("Augmented images written to " + outputFolder)
    
# end augment_continuous_sequence()
  
  
  
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
    images: stack of images [nBatch,width,height]
    masks: stack of corresponding masks [nBatch,width,height]
    indices: list of corresponding indices with the format ####_@@@@, where
        #### represents the parent image index, and @@@@ represents the
        augmentation index.
    indices_plus: indices with video number label appended as "_**"
OPTIONAL INPUTS:
    trainFraction: fraction of images as training (default 0.8)
RETURNS: 
    train_images: trainFraction of the input images
    train_masks: trainFraction of the input masks corresponding to train_images
    test_images: the other (1-trainFraction) of the input images
    test_masks: the other (1-trainFraction) of the input masks corresponding to test_images
    train_ids: list of ####_@@@@ format indices as above for the training set
    test_ids:  list of ####_@@@@ format indices as above for the test set
    train_ids_plus: train_ids wth video number label appended as "_**"
    test_ids_plus: test_ids with video number label appended as "_**"
'''
def train_test_split_noCheat(images, masks, indices, indices_plus, trainFraction=0.8):
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
    train_ids = np.array(indices)[trainIDList]
    test_ids = np.array(indices)[testIDList]
    train_ids_plus = np.array(indices_plus)[trainIDList]
    test_ids_plus = np.array(indices_plus)[testIDList]
    
    return train_images, train_masks, test_images, test_masks, train_ids, test_ids, train_ids_plus, test_ids_plus



'''
FUNCTION:
    nFolding() 
    
DESCRIPTION:
    Splits the input data into N sets, to roll for mutual training and testing.
    Done without "cheating", i.e. including augmented copies of the same
    original image in the same fold.
    
INPUTS: 
    N: number of folds to use
    images: stack of images [nBatch,width,height,nChannels]
    masks: stack of corresponding masks [nBatch,width,height]
    indices: list of corresponding indices with the format ####_@@@@, where
        #### represents the parent image index, and @@@@ represents the
        augmentation index.
    indices_plus: indices with video number label appended as "_**"
RETURNS: 
    image_folds: input images split into N folds [nFolds,nBatch,nWidth,nHeight]
    masks_folds: input masks matching image_folds [nFolds,nBatch,nWidth,nHeight]
    ids_folds:  list of ####_@@@@ format frame/aug indices for image_folds [nFolds, nBatch]
    ids_plus_folds: train_ids wth video number label appended as "_**" [nFolds, nBatch]
'''
def nFolding(N, images, masks, indices, indices_plus):
    
    # Generate a dictonary of the parent indices (frame numbers)
    parentList = {}
    for index,indexString in enumerate(indices):
        parentID = indexString[:4]
        if not parentID in parentList:
            parentList[parentID] = [] # initialize
        parentList[parentID].append(index) # add this index to the child list
    # endfor    
    
    # Divvy up the parent indices into folds
    nParents = len(parentList)
    parentIndices = [i for i in range(nParents)]
    random.shuffle(parentIndices)
    foldParentIndices = []
    parentsPerFold = int(nParents/N)
    for fold in range(N):
        myParentIndices = parentIndices[fold*parentsPerFold:(fold+1)*parentsPerFold] # omits any odd frames off the end
        foldParentIndices.append(myParentIndices)
    
    # Assign all the child images from each parent image to each fold per 
    # the above assignment.
    # Loop over folds
    foldIDLists = []
    for fold in range(N):
        i = 0
        thisFoldIDList = []
        # Loop over each frame index
        for parentID, children in parentList.items():
            # Loop over each input index belonging to this parent (frame)
            for childID in children: # indices of input arrays belonging to this frame
                if i in foldParentIndices[fold]: # if this frame goes in this fold
                    # Put this image idx into this fold
                    thisFoldIDList.append(childID)
            i += 1 # go to next parentID
        # endfor parent IDs
        foldIDLists.append(thisFoldIDList)
    # endfor folds
    
    # Put each image and mask into the appropriate set
    image_folds = []
    masks_folds = []
    ids_folds = []
    ids_plus_folds = []
    for fold in range(N):
        image_folds.append(images[foldIDLists[fold],:,:])
        masks_folds.append(masks[foldIDLists[fold],:,:])
        ids_folds.append(np.array(indices)[foldIDLists[fold]])
        ids_plus_folds.append(np.array(indices_plus)[foldIDLists[fold]])
    
    
    return image_folds, masks_folds, ids_folds, ids_plus_folds
# end nFolding

'''
FUNCTION:
    overlay_heatmap() 
    
DESCRIPTION:
    Takes the input heatmap, thresholds, cleans up, and finds edges.  Overlays
    edges as green on the input image and returns as a color image.
    
INPUTS: 
    heatmap: 2D array of target likelihood values (width,height)
    image: original image from which heatmap was calculated (width,height)
OPTIONAL INPUTS:
    heatThreshold: heatmap pixels greater than heatThreshold are deemed target
RETURNS: 
    The input image with the outlines of heatmap displayed in green
'''
def overlay_heatmap(heatmap, image, heatThreshold=0.5):
    # Massage heatmap into a thresholded binary array
    heatmap = 255*np.minimum(heatmap,np.ones(heatmap.shape))
    heatmap = heatmap.astype(np.uint8)
    binaryHeatmap = np.squeeze(heatmap) >= 255*heatThreshold
    binaryHeatmap = binaryHeatmap.astype(np.uint8)
    
    # Dilate and erode heatmap to remove little holes and spikes
    kernel2 = np.ones((2,2),np.uint8)
    dilated = cv2.dilate(binaryHeatmap,kernel2,iterations=1)
    eroded = cv2.erode(dilated,kernel2,iterations=1)
    
    # Dilate again by a 1-pixel larger kernel to get the outline of the 
    # above-threshold region
    kernel3 = np.ones((3,3),np.uint8)
    dilatedAgain = cv2.dilate(eroded,kernel3,iterations=1)
    outlines = (dilatedAgain-eroded) > 0
    
    # Make the outlines of the output image green
    colorImage = np.repeat(image[:,:,np.newaxis]*255,3,axis=2).astype(np.uint8)
    colorOutlines = np.zeros(colorImage.shape,np.uint8)
    colorOutlines[:,:,1] = outlines
    colorImage[colorOutlines>0] = 255
    
    return colorImage


'''
FUNCTION:
    overlay_heatmap_and_mask() 
    
DESCRIPTION:
    Takes the input heatmap, thresholds, cleans up, and finds edges.  Overlays
    edges as green on the input image. Then overlays the truth mask edges as 
    blue and returns as a color image
    
INPUTS: 
    heatmap: 2D array of target likelihood values (width,height)
    mask: 2D array of target truth values - binary (width,height)
    image: original image from which heatmap was calculated (width,height)
OPTIONAL INPUTS:
    heatThreshold: heatmap pixels greater than heatThreshold are deemed target
RETURNS: 
    The input image with the outlines of heatmap displayed in green and the 
    outlines of the truth mask displayed in blue.
'''
def overlay_heatmap_and_mask(heatmap, mask, image, heatThreshold=0.5):
    # Massage heatmap into a thresholded binary array
    heatmap = 255*np.minimum(heatmap,np.ones(heatmap.shape))
    heatmap = heatmap.astype(np.uint8)
    binaryHeatmap = np.squeeze(heatmap) > 255*heatThreshold
    binaryHeatmap = binaryHeatmap.astype(np.uint8)
    
    # Dilate and erode heatmap to remove little holes and spikes
    kernel2 = np.ones((2,2),np.uint8)
    dilated = cv2.dilate(binaryHeatmap,kernel2,iterations=1)
    eroded = cv2.erode(dilated,kernel2,iterations=1)
    
    # Dilate again by a 1-pixel larger kernel to get the outline of the 
    # above-threshold region
    kernel3 = np.ones((3,3),np.uint8)
    dilatedAgain = cv2.dilate(eroded,kernel3,iterations=1)
    outlines = (dilatedAgain-eroded) > 0
    
    # Dilate and erode mask in the same way
    binaryMask = mask == False
    binaryMask = binaryMask.astype(np.uint8)
    dilatedMask = cv2.dilate(binaryMask,kernel3,iterations=1)
    maskOutlines = (dilatedMask-binaryMask) > 0
    
    # Make the outlines of the heatmap green
    colorImage = np.repeat(image[:,:,np.newaxis]*255,3,axis=2).astype(np.uint8)
    colorOutlines = np.zeros(colorImage.shape,np.uint8)
    colorOutlines[:,:,1] = outlines
    colorImage[colorOutlines>0] = 255
    
    # Make the outlines of the mask blue
    colorMaskOutlines = np.zeros(colorImage.shape,np.uint8)
    colorMaskOutlines[:,:,0] = maskOutlines
    colorImage[colorMaskOutlines>0] = 255
    
    return colorImage

'''
FUNCTION:
    find_centerOfMass() 
    
DESCRIPTION:
    Finds the center of mass of the input heatmap (or mask) in the usual way.
    
INPUTS: 
    heatmap: 2D array of target likelihood values (width,height)
RETURNS: 
    The center of mass as a 2-element array, [xCOM,yCOM]
'''
def find_centerOfMass(heatmap):
    if len(heatmap.shape) != 2:
        heatmap = np.squeeze(heatmap)
    if len(heatmap.shape) != 2:
        print("Heatmap has wrong number of dimensions. Must be 2D:")
        print(heatmap.shape)
        
    totalWeight = np.sum(heatmap)
    
    if totalWeight == 0:
        # heatmap has no active pixels, return none
        return [None,None]
        
    width, height = heatmap.shape
          
    # Do COM calculation with matrices for speed        
    xx = np.repeat(np.expand_dims(np.arange(width ),axis=1),height,axis=1)
    yy = np.repeat(np.expand_dims(np.arange(height),axis=0),width, axis=0)
    xSum = np.sum(xx*heatmap)
    ySum = np.sum(yy*heatmap)
            
    centerOfMass = [xSum/totalWeight, ySum/totalWeight]
    
    return centerOfMass

'''
FUNCTION:
    overlay_point() 
    
DESCRIPTION:
    Overlays a point at the coordinates xyPoint onto the image. Point has the
    designated color and shape.  Point is injected into the image rather than
    plotted on top of it. Size is 5x5 pixels regardless of image size.
    
INPUTS: 
    image: 2D array to overlay point onto
    xyPoint: [x,y] coordinates of point to draw, in pixels
OPTIONAL INPUTS:
    mark: type of mark to draw. '+','x', or 's' (default '+')
    color: color of mark to draw. 'r','g', or 'b' (default 'g')
RETURNS: 
    Image with mark drawn over it in the designated color.
'''
def overlay_point(image,xyPoint,mark='+',color='g'):
    # If point is empty (happens if center of mass cannot be calculated), 
    # return original image.
    if xyPoint[0] == None or xyPoint[1] == None:
        return image
    
    # Colorize image if it is not yet colorized
    if len(image.shape) == 2:
        image = np.repeat(image[:,:,np.newaxis],3,axis=2)
        
    # Round xyPoint to nearest integer pixel and enforce boundaries
    for i,point in enumerate(xyPoint):
        if np.isnan(point):
            xyPoint[i] = 0.0
    xyPoint = [int(np.round(value)) for value in xyPoint]
    xyPoint = [np.min([np.max([2,val]),image.shape[i]-3]) for i,val in enumerate(xyPoint)]
       
    
    # Select mark type   
    if mark == '+':
        markXY = [[-2,0],[-1,0],[0,0],[1,0],[2,0],\
                    [0,-2],[0,-1],[0,1],[0,2]]
    elif mark == 'x':
        markXY = [[-2,-2],[-1,-1],[0,0],[1,1],[2,2], \
                  [-2,2],[-1,1],[1,-1],[2,-2]]
    elif mark == 's':
        markXY = [[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],[-1,2],[0,2],[1,2],[2,2],\
                  [2,1],[2,0],[2,-1],[2,-2],[1,-2],[0,-2],[-1,-2]]
    else:
        print("Mark type not recognized. Using plus sign")
        markXY = [[-2,0],[-1,0],[0,0],[1,0],[2,0],\
                  [0,-2],[0,-1],[0,1],[0,2]] 
        
    # Select color from 'r', 'g', 'b'
    if color == 'r':
        colorAs3 = [0,0,255]
    elif color == 'g':
        colorAs3 = [0,255,0]
    elif color == 'b':
        colorAs3 = [255,0,0]
    else:
        print("Color not recognized. Using green")
        colorAs3 = [0,255,0]

    # Overlay each pixel of the mark onto the image
    for x,y in markXY:
        image[xyPoint[0]+x,xyPoint[1]+y,:] = colorAs3

    return image

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
    # Regular-size injections
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48"),
                     iStart=175, iEnd=637, outShape=[48,64])
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48"),
                     iStart=175, iEnd=637, outShape=[48,64])
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab3","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab3","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48"),
                     iStart=175, iEnd=637, outShape=[48,64])
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab4","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab4","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48"),
                     iStart=175, iEnd=435, outShape=[48,64])
    '''                 
    '''
    # Baby injections
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1_baby","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1_baby","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_baby"),
                     iStart=175, iEnd=637, outShape=[48,64])
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2_baby","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2_baby","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_baby"),
                     iStart=175, iEnd=637, outShape=[48,64])
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab3_baby","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab3_baby","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_baby"),
                     iStart=175, iEnd=637, outShape=[48,64])
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab4_baby","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab4_baby","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_baby"),
                     iStart=175, iEnd=435, outShape=[48,64])
    '''    
    
    if not os.path.exists("augmentedContinuousSequences"):
        os.mkdir("augmentedContinuousSequences")
         
    # Continuous regular-sized injections
    augment_continuous_sequence(
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1","frame_"),
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1","mask_"),
         os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_mirror"),
         iStart=175, iEnd=637, outShape=[48,64])
    augment_continuous_sequence(
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2","frame_"),
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2","mask_"),
         os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_mirror"),
         iStart=175, iEnd=637, outShape=[48,64])
    augment_continuous_sequence(
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab3","frame_"),
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab3","mask_"),
         os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_mirror"),
         iStart=175, iEnd=637, outShape=[48,64])
    augment_continuous_sequence(
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab4","frame_"),
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab4","mask_"),
         os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_mirror"),
         iStart=175, iEnd=435, outShape=[48,64])
    
    # Continuous baby injections
    augment_continuous_sequence(
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1_baby","frame_"),
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab1_baby","mask_"),
         os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_baby_mirror"),
         iStart=175, iEnd=637, outShape=[48,64])
    augment_continuous_sequence(
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2_baby","frame_"),
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab2_baby","mask_"),
         os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_baby_mirror"),
         iStart=175, iEnd=637, outShape=[48,64])
    augment_continuous_sequence(
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab3_baby","frame_"),
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab3_baby","mask_"),
         os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_baby_mirror"),
         iStart=175, iEnd=637, outShape=[48,64])
    augment_continuous_sequence(
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab4_baby","frame_"),
         os.path.join("sequences","defaultGreenscreenVideo_over_roboticsLab4_baby","mask_"),
         os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_baby_mirror"),
         iStart=175, iEnd=435, outShape=[48,64])
         
