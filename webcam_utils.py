# -*- coding: utf-8 -*-
"""
FILE: webcam_utils
DESCRIPTION:
    Functions to read and stream images from USB webcam  

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""
import numpy as np
from imutils.video import WebcamVideoStream
import cv2
import time
from datetime import datetime
import os

# Using https://github.com/andfoy/textobjdetection/blob/master/ssd/demo/live.py as an example
def grabStream(folder='.',width=320,height=240,numFrames=100):
    # Initialize
    print("Initializing video stream")
    stream = WebcamVideoStream(src=0)
    #stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Start the video stream
    stream.start()
    time.sleep(1.0) # wait for stream to initialize
    video = np.zeros((numFrames,width,height))
    print("Video stream initialized")
    
    print("Capturing %d frames" % numFrames)
    start_time = datetime.now()
    #while True:
    for i in range(numFrames):
        video[i,:,:] = stream.read() # grab next frame
    end_time = datetime.now()  
    timeElapsed = (end_time-start_time).total_seconds()
    
    print("%d frames captured in %g seconds: framerate = %g Hz" % 
        (numFrames,timeElapsed,numFrames/timeElapsed))
        
    # Save out at the end   
    print("Saving frames to " + folder) 
    for i in range(numFrames):
        filestr = "frame_%04d" % i
        fullpath = os.path.join(folder, filestr+'.jpg')
        cv2.imwrite(np.squeeze(video[i,:,:]),fullpath)

    print("Complete!")

if __name__ == "__main__":
    grabStream()
