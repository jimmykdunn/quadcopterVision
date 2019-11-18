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

import cv2
import os
import numpy as np
import videoUtilities as vu
import shutil
from datetime import datetime
import kalman
import matplotlib.pyplot as plt

try:
    import webcam_utils as wcam
except:
    pass

def run(modelPath, nnFramesize=(64,48), save=False, folder='webcam',
        showHeatmap=False, liveFeed=True, displayScale=1, USE_KALMAN=True,
        filestream=None, largeDisplay=False, heatmapThresh=0.75, runAlgorithm=True):
    # Import the trained neural network
    print("Loading saved neural network from " + modelPath+'.pb')
    tensorflowNet = cv2.dnn.readNetFromTensorflow(modelPath+'.pb')
    print("Neural network sucessfully loaded")
    
    # Prepare the output folder
    if save:
        if os.path.exists(folder): # only rm if it exists
            shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            os.mkdir(folder)
            
    # Initialize COM history vectors
    history_rawCOM = np.array([[],[]])
    history_kalmanCOM = np.array([[],[]])
        
    # Initialize the video stream from the camera
    if filestream == None:
        webcam = wcam.videoStream()
        
        print("Video stream started")
        print("Press P key to pause/play")
        print("Press ESC key to quit")
    else:
        print("Reading image sequence from " + filestream)
        frameset = vu.pull_sequence(filestream)
        print("Image sequence shape:")
        print(frameset.shape)
    
    # Continuously pull and display frames from the camera until stopped
    i=0
    start_time = datetime.now()
    prev_time = start_time
    while True:
        if filestream == None:
            # Pull from camera   
            frame = webcam.grabFrame() # grab a frame
            
        else:
            # Break off and exit if past end of sequence
            if i >= frameset.shape[0]:
                break
                
            # Pull from file
            frame = np.repeat(frameset[i,:,:,:],3,axis=2)
            
        if i == 0:
            print("Raw frame shape: " + str(frame.shape))
            
                   
        curr_time = datetime.now()
        
        # If statement around all the processing. Always true - only set to false
        # for latency comparison tests.
        if runAlgorithm:
        
            # Massage frame to be the right size and colorset#
            nnFrame = cv2.resize(frame,nnFramesize)
            nnFrame = nnFrame[:,:,0] * float(1.0/255.0)
            nnFrame = np.squeeze(nnFrame)
            
            # Initialize Kalman filter as motionless at center of image if this
            # is the first frame. Uncertainty is the size of the image.
            if i == 0:
                kalmanFilter = kalman.kalman_filter(
                    nnFrame.shape[0]/2,nnFrame.shape[1]/2, # x,y
                    0,0,                                   # vx, vy
                    nnFrame.shape[0]/2,nnFrame.shape[1]/2, # sigX, sigY
                    0,0)                                   # sigVX, sigVY
                # Initialize useful arrays for later
                allZeros = np.zeros_like(nnFrame)
                all255  = np.ones_like(nnFrame)*255
            
            # Execute a forward pass of the neural network on the frame to get a
            # heatmap of target likelihood
            # This is now by far the limiting temporal factor - without it the 
            # framerate is in the 300Hz range, with it framerate is in the 50Hz range.
            tensorflowNet.setInput(nnFrame)
            heatmap = tensorflowNet.forward()
            heatmap = np.squeeze(heatmap)
            
            # Optionally resize everything to be larger for display.  Has the 
            # drawback of increased latency
            scale = 1
            if largeDisplay:
                scale = 4
                bigShape = (int(heatmap.shape[1]*scale), int(heatmap.shape[0]*scale))
                heatmap = cv2.resize(heatmap, bigShape)
                nnFrame = cv2.resize(frame[:,:,0], bigShape) * float(1.0/255.0)
                
            #print("Heatmap big shape (%d,%d)" % heatmap.shape[:2])
            #print("Frame big shape (%d,%d)" % nnFrame.shape[:2])
            
            if i == 0:
                # Initialize useful arrays for later
                allZeros = np.zeros_like(nnFrame)
                all255  = np.ones_like(nnFrame)*255
            
            
            # Overlay heatmap contours onto the image (speed negligible)
            #print(np.min(heatmap), np.max(heatmap))
            overlaidNN = vu.overlay_heatmap(heatmap, nnFrame, heatThreshold=0.75, scale=scale)
            heatmap = heatmap*255.0 # scale appropriately
            
            # Find the center of mass for this frame
            heatmapCOM = vu.find_centerOfMass(heatmap, minThresh=heatmapThresh*255)
            targetVisible = True
            if heatmapCOM == [None, None]:
                targetVisible = False
                print("Target not found")
                heatmapCOM = [heatmap.shape[0]/2,heatmap.shape[1]/2] # default to center of image
            print("Frame %04d" % i) 
            print("    Pre-kalman:  %02d, %02d" % (heatmapCOM[0], heatmapCOM[1]))
            history_rawCOM = np.append(history_rawCOM,np.expand_dims(heatmapCOM,1),axis=1)
            
            # Apply Kalman filter to the COM centroid measurement if desired
            if USE_KALMAN:
                kalmanFilter.project((curr_time-prev_time).total_seconds())
                
                # Only update the kalman filter when we actually detect the target
                if targetVisible:
                    # Massage the heatmap and calculate average per-pixel energy
                    heatmapClip = np.maximum(np.minimum(heatmap,all255),allZeros) # range is 0 to 255
                    heatmapClip = heatmapClip.astype(np.float32)
                    heatmapClip[heatmapClip < 255.0*heatmapThresh] = 0.0
                    heatmapMeanEnergy = np.mean(heatmapClip)/255.0 # range is 0 to 1
                    print("    Heatmap mean energy: %g" % heatmapMeanEnergy)
                                
                    # Calculate the measurement error as the inverse of total heatmap 
                    # energy becuase more energy = better localization accuracy
                    # 0.08 here is an empirically determined factor to get the 
                    # uncertanties in the expected range. 0.01 just prevents infinities.
                    sigX = 0.08 * (1.0/scale) * 0.5 * nnFrame.shape[0] / (heatmapMeanEnergy + 0.01)
                    sigY = 0.08 * (1.0/scale) * 0.5 * nnFrame.shape[1] / (heatmapMeanEnergy + 0.01)
                    print("    (sigX, sigY) = (%g,%g)" % (sigX,sigY))
                    
                    # Update the kalman filter with the measured COM and measurement error
                    kalmanFilter.update(heatmapCOM, [[sigX, 0],[0, sigY]])
                
                # Pull the COM from the kalman vector state
                heatmapCOM = kalmanFilter.stateVector[:2]
            # endif USE_KALMAN
            
            # Overlay the target location
            overlaidNN = vu.overlay_point(overlaidNN,heatmapCOM,color='g')
            print("    Post-kalman: %02d, %02d" % (heatmapCOM[0], heatmapCOM[1]))
            print('')
            history_kalmanCOM = np.append(history_kalmanCOM,np.expand_dims(heatmapCOM,1),axis=1)
            
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
        # end if runAlgorithm
        
        # keypress control
        key = cv2.waitKey(1) & 0xFF
        
        # Display video feed live if desired
        if liveFeed:
            if key == ord('p'): # pause
                while True:
                    key2 = cv2.waitKey(1) or 0xff
                    if runAlgorithm:
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
            if runAlgorithm:
                cv2.imshow('frame',displayThis)
        # end if liveFeed
       
        if runAlgorithm:     
            # Save each frame if desired
            if save:
                # Save raw frame
                filestr = "frameRaw_%04d.jpg" % i
                fullpath = os.path.join(folder, filestr)
                cv2.imwrite(fullpath, nnFrame*255.0)
                
                # Save displayed frame (with heatmap overlay)
                filestr = "frameDisplay_%04d.jpg" % i
                fullpath = os.path.join(folder, filestr)
                cv2.imwrite(fullpath, displayThis)
            
        # end if runAlgorithm  
        
        # Always show first frame so that exit key works  
        if i==1:
            cv2.imshow('frame',frame) 
            
            
        # End on Esc keypress   
        if key == 27: 
            break
        
        # Increment frame loop counter
        i+=1
        
        # Save off time for use by kalman filter
        prev_time = curr_time
    # end while True
    
    # Calculate framerate statistics
    end_time = datetime.now()
    timeElapsed = (end_time-start_time).total_seconds()
    print("%d frames captured in %g seconds: framerate = %g Hz" % 
        (i,timeElapsed,i/timeElapsed))
    
    if runAlgorithm:
        # Print directory frames are saved to if they are being saved
        if save:
            print("Wrote image pairs to " + folder + ' with shape ' +
                str(displayThis.shape))
                
                
        # Make a final display of the snail trail of the COM
        xCoords = history_rawCOM[1,:]
        yCoords = heatmap.shape[0] - history_rawCOM[0,:] # invert y axis for consistency
        plt.plot(xCoords, yCoords, 'k+', label="raw")
        xCoordsKalman = history_kalmanCOM[1,:]
        yCoordsKalman = heatmap.shape[0] - history_kalmanCOM[0,:] # invert y axis for consistency
        plt.plot(xCoordsKalman, yCoordsKalman, 'go', label="Kalman filtered")
        plt.axis('equal')
        plt.xlim([0,nnFrame.shape[1]])
        plt.ylim([0,nnFrame.shape[0]])
        plt.xlabel('Horizontal pixels')
        plt.ylabel('Vertical pixels')
        plt.title('Quadrotor centroid history')
        plt.legend()
        plt.savefig("snailTrail.png")
        print("Wrote snail trail to snailTrail.png")
        plt.show()
        #print(history_rawCOM.shape)
    
    # end if runAlgorithm
            
    # Cleanup
    print("Cleaning up")
    if filestream == None:
        webcam.stop()
    cv2.destroyAllWindows()
    print("Done")
    

# Run if called directly
if __name__ == "__main__":
    # Run from a saved stream
    imgBase = os.path.join('webcamSaves','webcam_square','frameRaw_')
    #run(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_mirror_60k_sW00p50_1M00p00_2M00p00_49k'),
    #   save=True, liveFeed=True, showHeatmap=True, USE_KALMAN=True, filestream=imgBase)
    #run(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_mirror_60k_sW00p50_1M00p00_2M00p00_49k'),
    #    save=True, liveFeed=True, showHeatmap=True, USE_KALMAN=True, filestream=imgBase, heatmapThresh=0.5)
    #run(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_biasAdd_sW00p50_fold3'),
    #    save=True, liveFeed=True, showHeatmap=True, USE_KALMAN=True, filestream=imgBase, heatmapThresh=0.5)

    # Run from a live camera stream
    run(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_biasAdd_sW00p50_fold0'),
        save=True, liveFeed=True, showHeatmap=True, USE_KALMAN=True, largeDisplay=False, heatmapThresh=0.5)
    #run(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_mirror_60k_sW00p00_1M00p00_2M00p00_16k'),
    #    save=True, liveFeed=True, showHeatmap=True, USE_KALMAN=True, largeDisplay=True, heatmapThresh=0.5)

    # Run a latency test by turning off the live displays and saving like we would do in real time
    # for optimization.
    #run(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_biasAdd_sW00p50_fold0'),
    #    save=False, liveFeed=False, showHeatmap=False, USE_KALMAN=True, largeDisplay=False, heatmapThresh=0.5)
    #run(os.path.join('homebrew_hourglass_nn_save_GOOD','modelFinal_full_biasAdd_sW00p50_fold0'),
    #    save=False, liveFeed=False, showHeatmap=False, USE_KALMAN=True, largeDisplay=False, heatmapThresh=0.5,
    #    runAlgorithm=False)
    


