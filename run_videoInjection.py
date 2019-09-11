# -*- coding: utf-8 -*-
"""
FUNCTION: run_videoInjection
DESCRIPTION:
    Reads in a greenscreened video and a background video and puts the 
    background into the greenscreened video. Greenscreened video and
    background video are assumed to have the same framerate, resolution, 
    and duration. 
    
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
    
    vidExt = '.mp4'
    framerate = 30 # frames per second
    
    # Form the filename of the output video
    greenVideoName = os.path.splitext(greenVideoFile)[0]
    backgroundVideoName = os.path.splitext(backgroundVideoFile)[0]
    outVideoFile = greenVideoName + '_over_' + backgroundVideoName + vidExt
    
    # Open input video streams
    greenVideoStream = cv2.VideoCapture(greenVideoName)
    backgroundVideoStream = cv2.VideoCapture(backgroundVideoName)
    assert greenVideoStream.isOpened()
    assert backgroundVideoStream.isOpened()
    
    # Determine size of each video
    greenWidth,greenHeight = \
        greenVideoStream.get(3), greenVideoStream.get(4)
    backgroundWidth, backgroundHeight = \
        backgroundVideoStream.get(3), backgroundVideoStream.get(4)
    assert greenWidth == backgroundWidth
    assert greenHeight == backgroundHeight
    
    # Open output video stream
    outStream = cv2.VideoWriter(outVideoFile,cv2.VideoWriter_fourcc(*'DIVX'), \
                                framerate, [greenWidth, greenHeight])
    
    # Read in the greenscreen video and process frame-by-frame
    success_g, success_b = True, True
    while success_g and success_b:
        # Read a frame of the each video
        success_g, greenFrame = greenVideoStream.read()
        success_b, backgroundFrame = backgroundVideoStream.read()
        
        # Inject the background
        frame = injectBackground(greenFrame, backgroundFrame)
        
        # Write frame to output video stream
        outStream.write(frame)
        
    # Save it off so we can play the video
    outStream.release()
    
# end run_videoInjection()

# Run with defaults if at highest level
if __name__ == "__main__":
    run_videoInjection("defaultGreenscreenVideo.mp4", "defaultBackgroundVideo.mp4")