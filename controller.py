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
import shutil
from datetime import datetime

def run(modelPath, nnFramesize=(64,64), save=False, folder='webcam',
        showHeatmap=False, liveFeed=True, displayScale=1):
    # Import the trained neural network
    print("Loading saved neural network from " + modelPath+'.pb')
    tensorflowNet = cv2.dnn.readNet(modelPath+'.pb')
    print("Neural network sucessfully loaded")
    
    # Prepare the output folder
    if save:
        if os.path.exists(folder): # only rm if it exists
            shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        
    # Initialize the video stream from the camera
    webcam = wcam.videoStream()
    
    print("Video stream started")
    print("Press P key to pause/play")
    print("Press ESC key to quit")
    
    # Continuously pull and display frames from the camera until stopped
    i=0
    start_time = datetime.now()
    while True:
        frame = webcam.grabFrame() # grab a frame
        if i == 0:
            print("Raw frame shape: " + str(frame.shape))
        
        # Massage frame to be the right size and colorset#
        nnFrame = cv2.resize(frame,nnFramesize)
        nnFrame = nnFrame[:,:,0] * float(1.0/255.0)
        nnFrame = np.squeeze(nnFrame)
        
        # Execute a forward pass of the neural network on the frame to get a
        # heatmap of target likelihood
        # This is now by far the limiting temporal factor - without it the 
        # framerate is in the 300Hz range, with it framerate is in the 50Hz range.
        tensorflowNet.setInput(nnFrame)
        heatmap = tensorflowNet.forward()
        heatmap = np.squeeze(heatmap)*255.0 # scale appropriately
        
        # Overlay center of mass and heatmap contours onto the image (speed negligible)
        overlaidNN = vu.overlay_heatmap(heatmap, nnFrame, heatThreshold=0.5)
        heatmapCOM = vu.find_centerOfMass(heatmap)
        overlaidNN = vu.overlay_point(overlaidNN,heatmapCOM,color='g')
        
        if showHeatmap:
            # Display the heatmap and the image side-by-side
            heatmap = np.minimum(heatmap,np.ones(heatmap.shape)*255)
            heatmap = np.maximum(heatmap,np.zeros(heatmap.shape))
            heatmap = heatmap.astype(np.uint8) # map to uint8 to match nnFrame
            heatmapColor = np.repeat(heatmap[:,:,np.newaxis],3,axis=2)
            displayThis = np.concatenate([overlaidNN, heatmapColor],axis=0)
        else:
            # Just display image with outlines and COM
            displayThis = overlaidNN
        
        # keypress control
        key = cv2.waitKey(1) & 0xFF
        
        # Display video feed live if desired
        if liveFeed:
            if key == ord('p'): # pause
                while True:
                    key2 = cv2.waitKey(1) or 0xff
                    cv2.imshow('frame', displayThis) # show last grabbed frame
                    if key2 == ord('p'): # resume
                        break
            # end if paused
            
            # Enlarge display for ease of viewing if desired
            if displayScale != 1:
                pairFactor = 1
                if showHeatmap:
                    pairFactor = 2
                pair = cv2.resize(displayThis,(nnFramesize[0]*displayScale,
                    nnFramesize[1]*displayScale*pairFactor))
            # end if displayScale != 1
                    
            # Show the current frame with neural net mask
            cv2.imshow('frame',displayThis)
        # end if liveFeed
        
        # Save each frame if desired
        if save:
            filestr = "frame_%04d.jpg" % i
            fullpath = os.path.join(folder, filestr)
            cv2.imwrite(fullpath, displayThis)
        
        # Always show first frame so that exit key works  
        if i==1:
            cv2.imshow('frame',displayThis)   
            
        # End on Esc keypress   
        if key == 27: 
            break
        
        # Increment frame loop counter
        i+=1
    # end while True
    
    # Calculate framerate statistics
    end_time = datetime.now()
    timeElapsed = (end_time-start_time).total_seconds()
    print("%d frames captured in %g seconds: framerate = %g Hz" % 
        (i,timeElapsed,i/timeElapsed))
    
    # Print directory frames are saved to if they are being saved
    if save:
        print("Wrote image pairs to " + folder + ' with shape ' +
            str(displayThis.shape))
            
    # Cleanup
    print("Cleaning up")
    webcam.stop()
    cv2.destroyAllWindows()
    print("Done")
    

# Run if called directly
if __name__ == "__main__":
    run(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full'),
        save=True, liveFeed=True)
