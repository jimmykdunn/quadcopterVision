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
 

"""
Class: videoStream
DESCRIPTION:
    Basic class for starting and capturing frames from a video stream. 
    Ultimately serves as a wrapper for imutils.video.WebcamVideoStream.

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
    
REFERENCES:
   Used https://github.com/andfoy/textobjdetection/blob/master/ssd/demo/live.py
   as an example.
""" 
class videoStream:
    def __init__(self):
        print("Initializing video stream")
        self.stream = WebcamVideoStream(src=0)
        
        # Start the video stream
        self.stream.start()
        time.sleep(1.0) # wait for stream to initialize
        print("Video stream initialized")
    # end init
    
    def grabFrame(self):
        return self.stream.read() # grab a frame and return
    # end grabFrame  
    
    def stop(self):
        self.stream.stop()  
    
# end class videoStream
 
"""
grabStream()
DESCRIPTION:
    Simple webcam test - grabs numFrames from webcam and saves them to the 
    designated folder as jpg images.
    
OPTIONAL INPUTS:
    folder: directory to write the captured frames to
    numFrames: total number of frames to grab
    
OUTPUTS: 
    Saves the captured frames to file

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
    
REFERENCES:
   Used https://github.com/andfoy/textobjdetection/blob/master/ssd/demo/live.py
   as an example.
"""
def grabStream(folder='.',numFrames=100):
    # Initialize
    stream = videoStream()
    
    # Create the output folder if it does not yet exist
    if not os.path.exists(folder):
        os.mkdir(folder)
    print("Saving frames to " + folder) 
    
    # Loop over the number of desired frames at maximum speed
    print("Capturing %d frames" % numFrames)
    start_time = datetime.now()
    for i in range(numFrames):
        frame = stream.grabFrame() # grab next frame
        
        # Save the frame to file
        filestr = "frame_%04d.jpg" % i
        fullpath = os.path.join(folder, filestr)
        cv2.imwrite(fullpath, frame)
        print("Wrote " + fullpath + ' ' + str(frame.shape))
    # end capture
    stream.stop()
        
    # Calculate framerate
    end_time = datetime.now()  
    timeElapsed = (end_time-start_time).total_seconds()
    
    print("%d frames captured in %g seconds: framerate = %g Hz" % 
        (numFrames,timeElapsed,numFrames/timeElapsed))

    print("Complete!")
# end grabStream

# Run grabStream if called directly
if __name__ == "__main__":
    grabStream(folder='webcam')
