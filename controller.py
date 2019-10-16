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
import os
import numpy as np
import videoUtilities as vu

def run(modelPath, nnFramesize=(64,64)):
    # Import the trained neural network
    print("Loading saved neural network from " + modelPath+'.pb')
    tensorflowNet = cv2.dnn.readNet(modelPath+'.pb')
    print("Neural network sucessfully loaded")

    # Initialize the video stream from the camera
    webcam = wcam.videoStream()
    
    print("Video stream started")
    print("Press P key to pause/play")
    print("Press ESC key to quit")
    
    # Continuously pull and display frames from the camera until stopped
    while True:
        frame = webcam.grabFrame() # grab a frame
        
        # Massage frame to be the right size and colorset
        nnFrameLargeColor = np.mean(frame,axis=2)/float(255)
        nnFrame = cv2.resize(nnFrameLargeColor,nnFramesize)
        nnFrame = np.squeeze(nnFrame)
        
        # Execute a forward pass of the neural network on the frame to get a
        # heatmap of target likelihood
        tensorflowNet.setInput(nnFrame)
        heatmap = tensorflowNet.forward()
        heatmap = np.squeeze(heatmap)*255.0 # scale appropriately
        
        # Overlay center of mass and heatmap contours onto the image
        overlaidNN = vu.overlay_heatmap(heatmap, nnFrame, heatThreshold=0.5)
        heatmapCOM = vu.find_centerOfMass(heatmap)
        overlaidNN = vu.overlay_point(overlaidNN,heatmapCOM,color='g')
        
        # Display the heatmap and the image side-by-side
        heatmap = np.minimum(heatmap,np.ones(heatmap.shape)*255)
        heatmap = np.maximum(heatmap,np.zeros(heatmap.shape))
        heatmap = heatmap.astype(np.uint8) # map to uint8 to match nnFrame
        heatmapColor = np.repeat(heatmap[:,:,np.newaxis],3,axis=2)
        pair = np.concatenate([overlaidNN, heatmapColor],axis=0)
        
        # keypress control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'): # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', pair) # show last grabbed frame
                if key2 == ord('p'): # resume
                    break
        
        # Show the current frame with neural net mask
        pair = cv2.resize(pair,(nnFramesize[0]*4,nnFramesize[1]*4*2)) # larger for ease of viewing
        cv2.imshow('frame',pair)
        
        if key == 27: # exit (Esc)
            break
            
    # Cleanup
    print("Cleaning up")
    webcam.stop()
    cv2.destroyAllWindows()
    print("Done")
    

# Run if called directly
if __name__ == "__main__":
    run(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full'))
