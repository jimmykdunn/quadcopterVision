# -*- coding: utf-8 -*-
"""
FILE: controller
DESCRIPTION:
    Runs the UAV camera, ingests and processes images, displays a live video
    feed, and calculates feedback controls. 

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""

import webcam_utils as wcam
import cv2

def run(nnFramesize=(64,64)):
    # Import the trained neural network
    

    # Initialize the video stream from the camera
    webcam = wcam.videoStream()
    
    print("Video stream started")
    print("Press P key to pause/play")
    print("Press ESC key to quit")
    
    # Continuously pull and display frames from the camera until stopped
    while True:
        frame = webcam.grabFrame() # grab a frame
        
        # Massage frame to be the right size and colorset
        nnFrameLargeColor = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        nnFrame = cv2.resize(nnFrameLargeColor,nnFramesize)
        
        # Execute a forward pass of the neural network on the frame to get a
        # heatmap of target likelihood
        
        # Display the heatmap and the image side-by-side
        
        # keypress control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'): # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', nnFrame) # show last grabbed frame
                if key2 == ord('p'): # resume
                    break
        
        # Show the current frame with neural net mask
        cv2.imshow('frame',nnFrame)
        
        if key == 27: # exit (Esc)
            break
            
    # Cleanup
    print("Cleaning up")
    webcam.stop()
    cv2.destroyAllWindows()
    print("Done")
    

# Run if called directly
if __name__ == "__main__":
    run()
