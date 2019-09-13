# -*- coding: utf-8 -*-
"""
FUNCTION: run_videoInjection
DESCRIPTION:
    Reads in a greenscreened video and a background video and puts the 
    background into the greenscreened video. Greenscreened video and
    background video are assumed to have the same framerate and resolution.
    Duration of the blended video will match the shorter of the two videos.
    
INPUTS: 
    greenVideoFile: location of a greenscreened video file readable by opencv
    backgroundVideoFile: location of a background video file readable by opencv
    
OUTPUTS: 
    Video with greenscreen pixels replaced with background pixels. File name
    is the catenation of the greenVideoFile and backgroundVideoFile filenames.

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""

import injectBackground
import cv2
import os

def run_videoInjection(greenVideoFile, backgroundVideoFile):
    
    vidExt = '.avi'
    framerate = 30 # frames per second
    
    # Form the filename of the output video
    greenVideoName = os.path.splitext(greenVideoFile)[0]
    backgroundVideoName = os.path.splitext(backgroundVideoFile)[0]
    outVideoFile = greenVideoName + '_over_' + backgroundVideoName + vidExt
    
    # Open input video streams
    greenVideoStream = cv2.VideoCapture(greenVideoFile)
    backgroundVideoStream = cv2.VideoCapture(backgroundVideoFile)
    assert greenVideoStream.isOpened()
    assert backgroundVideoStream.isOpened()
    
    # Determine size of each video
    greenWidth,greenHeight = \
        int(greenVideoStream.get(3)), int(greenVideoStream.get(4))
    backgroundWidth, backgroundHeight = \
        int(backgroundVideoStream.get(3)), int(backgroundVideoStream.get(4))
    assert greenWidth == backgroundWidth
    assert greenHeight == backgroundHeight
    
    # Open output video stream
    outStream = cv2.VideoWriter(outVideoFile,cv2.VideoWriter_fourcc('X','V','I','D'), \
                                framerate, (greenWidth, greenHeight))
    
    # Read in the greenscreen video and process frame-by-frame
    success_g, success_b = True, True
    iFrame = 0
    while success_g and success_b:
        # Read a frame of the each video
        success_g, greenFrame = greenVideoStream.read()
        success_b, backgroundFrame = backgroundVideoStream.read()
        if not success_g or not success_b:
            break
        
        # Inject the background
        frame = injectBackground.injectBackground(greenFrame, backgroundFrame)
        
        # Write frame to output video stream
        outStream.write(frame)
        
        print('Injecting onto frame ' + str(iFrame))
        iFrame += 1
        
    # Save it off so we can play the video
    outStream.release()
    
# end run_videoInjection()

# Run with defaults if at highest level
if __name__ == "__main__":
    run_videoInjection("defaultGreenscreenVideo.avi", "defaultBackgroundVideo.avi")