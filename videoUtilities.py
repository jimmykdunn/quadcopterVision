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
        outStream.release()
    
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
    
RETURNS: 
    Video from inFile as a [nframes,nx,ny,nColors] 4D array
"""
def pull_sequence(inFileBase, ext='.jpg', iStart=0, iEnd=-1, color=False):
    index = iStart
    
    # Read first frame to get sizes
    indexStr = "%04d" % index
    fullFilename = inFileBase + indexStr + ext
    if os.path.isfile(fullFilename):
        frame = cv2.imread(fullFilename)
        if not color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #frame = frame[:,:,0] # extract red channel
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
    
    

    
# Testing here
if __name__ == "__main__":
    myVideo = pull_sequence(os.path.join("sequences",
        "defaultGreenscreenVideo_over_PHO_hallway","frame_"), iStart=330, iEnd=360)
    print("nFrames: %d" % myVideo.shape[0])
    
    plt.imshow(np.squeeze(myVideo[0,:,:,:]),cmap='Greys')
    plt.show()
    
    
