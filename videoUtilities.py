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
    
    outVideo = np.zeros([None,width,height,3])
       
    try:
        # Read in the video frame-by-frame
        success = True
        iFrame = 0
        while success:
            # Read a frame
            success, frame = inStream.read()
                
            if not success:
                break
           
            outVideo = np.concatenate(outVideo,frame,axis=0)
            
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
        
    print("Video dimensions: (%d,%d)" % (width, height))
    
    # Initialize output video array
    outVideo = np.zeros([1,width,height,nColors])
    outVideo[0,:,:,:] = frame
    index += 1
    print("Outvideo dimensions:", outVideo.shape)
    
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
    
    # Make output directory if it does not yet exist
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    
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
        if len(rawImage.shape) == 1:
            if rawImage == -1 or rawMask == -1:
                break # we reached the end or had problems reading
            
        iAugmentation = 0
        
        # Apply some random crops and resizes to the raw images and masks, 
        # always applying the same crop and resize to the image and its
        # corresponding mask.
        nbRandomCropResize = 8
        lowScl = np.max(np.divide(outShape,rawImage.shape[1:3])) # don't go smaller than outimage size
        scaleSet = np.random.uniform(low=lowScl, high=lowScl*2, size=nbRandomCropResize)
        outSizeSet = [[np.ceil(dim * scale).astype(int) for dim in rawImage.shape[2:0:-1]] for scale in scaleSet]
        for iCropResize in range(nbRandomCropResize):
            resizedImage = cv2.resize(np.squeeze(rawImage), tuple(outSizeSet[iCropResize]))
            croppedImage = resizedImage[:outShape[0],:outShape[1]]
            finalAugmentedImage = croppedImage
            resizedMask = cv2.resize(np.squeeze(rawMask), tuple(outSizeSet[iCropResize]))
            croppedMask = resizedMask[:outShape[0],:outShape[1]]
            finalAugmentedMask = croppedMask
            
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
    
# end augment_sequence()
    
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
    augment_sequence(os.path.join("sequences","defaultGreenscreenVideo_over_PHO_hallway","frame_"),
                     os.path.join("sequences","defaultGreenscreenVideo_over_PHO_hallway","mask_"),
                     os.path.join("augmentedSequences","defaultGreenscreenVideo_over_PHO_hallway"),
                     iStart=330, iEnd=332)
    
