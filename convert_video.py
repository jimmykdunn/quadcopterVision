

# -*- coding: utf-8 -*-
"""
FUNCTION: convert_video
DESCRIPTION:
    Uses cv2 to convert a video in the input format to the chosen output format
    
INPUTS: 
    inFile: movie file readable by cv2
    outFile: movie file writeable by cv2
    
OUTPUTS: 
    Video from inFile in the format of outFile

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""

import cv2
import os

def convert_video(inFile, outFile):
    
    framerate = 30
    
    # Open input video streams
    inStream = cv2.VideoCapture(inFile)
    assert inStream.isOpened()
    width,height = int(inStream.get(3)), int(inStream.get(4))
    
    
    # Open output video stream
    outStream = cv2.VideoWriter(outFile,cv2.VideoWriter_fourcc('X','V','I','D'), \
                                framerate, (width, height))
    
    try:
        # Read in the video frame-by-frame
        success = True
        iFrame = 0
        while success:
            # Read a frame
            success, frame = inStream.read()
                
            if not success:
                break
           
            # Write frame to output video stream
            outStream.write(frame)
            
            print('Converting frame ' + str(iFrame))
            iFrame += 1
        # while success
        
    finally:       
        # Save it off so we can play the video
        print("Exiting cleanly")
        outStream.release()
    
# end convert_video()

# Run with defaults if at highest level
if __name__ == "__main__":
    convert_video(os.path.join('backgroundVideos','roboticsLab2.MOV'),
                  os.path.join('backgroundVideos','roboticsLab2.avi'))
    
