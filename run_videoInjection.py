

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
    bgRotate_deg: degrees CCW to rotate the background by (0, 90, 180, or 270)
    
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

import greenscreenTools as gst
import cv2
import os
import struct
import numpy as np

def run_videoInjection(greenVideoFile, backgroundVideoFile, bgRotate_deg=0):
    
    vidExt = '.avi'
    framerate = 30 # frames per second
    
    # Form the filename of the output video
    greenVideoName = os.path.splitext(greenVideoFile)[0]
    backgroundVideoName = os.path.splitext(backgroundVideoFile)[0].split(os.sep)[-1]
    outVideoBase = greenVideoName + '_over_' + backgroundVideoName
    outVideoFile = outVideoBase + vidExt
    outMaskBase = greenVideoName + '_mask'
    outMaskFile = outMaskBase + vidExt
    
    # Open input video streams
    print("Opening input video streams")
    inputExtension = os.path.splitext(greenVideoFile)[1]
    if 'avi' in inputExtension:
        greenVideoStream = cv2.VideoCapture(greenVideoFile)
        print("Opened "+greenVideoFile)
        backgroundVideoStream = cv2.VideoCapture(backgroundVideoFile)
        print("Opened "+backgroundVideoFile)
        assert greenVideoStream.isOpened()
        assert backgroundVideoStream.isOpened()
        greenWidth,greenHeight = \
            int(greenVideoStream.get(3)), int(greenVideoStream.get(4))
        backgroundWidth, backgroundHeight = \
            int(backgroundVideoStream.get(3)), int(backgroundVideoStream.get(4))
    else:
        greenVideoStream = open(greenVideoFile,'rb')  
        print("Opened "+greenVideoFile)
        backgroundVideoStream = open(backgroundVideoFile,'rb') 
        print("Opened "+backgroundVideoFile) 
        greenWidth = struct.unpack('i',greenVideoStream.read(4))[0]
        greenHeight = struct.unpack('i',greenVideoStream.read(4))[0]
        greenNColors = struct.unpack('i',greenVideoStream.read(4))[0]
        backgroundWidth = struct.unpack('i',backgroundVideoStream.read(4))[0]
        backgroundHeight = struct.unpack('i',backgroundVideoStream.read(4))[0]
        backgroundNColors = struct.unpack('i',backgroundVideoStream.read(4))[0]
    
    
    # Open output video stream
    if bgRotate_deg == 90 or bgRotate_deg == 270:
        outWidth = backgroundHeight
        outHeight = backgroundWidth
    else:
        outWidth = backgroundWidth
        outHeight = backgroundHeight
    outStream = cv2.VideoWriter(outVideoFile,cv2.VideoWriter_fourcc('X','V','I','D'), \
                                framerate, (outWidth, outHeight))
    print("Opened "+outVideoFile+" for output video stream") 
    
    # Open output mask video stream
    outMaskStream = cv2.VideoWriter(outMaskFile,cv2.VideoWriter_fourcc('X','V','I','D'), \
                                framerate, (outWidth, outHeight))
    print("Opened "+outMaskFile+" for output mask stream") 
    
    try:
        # Read in the greenscreen video and process frame-by-frame
        success_g, success_b = True, True
        iFrame = 0
        while success_g and success_b:
            # Read a frame of each video
            if 'avi' in inputExtension:
                success_g, greenFrame = greenVideoStream.read()
                success_b, backgroundFrame = backgroundVideoStream.read()
            else:
                greenFrame = np.zeros([greenWidth*greenHeight*greenNColors,1]).astype(np.uint8)
                greenFrameRaw = greenVideoStream.read(greenWidth*greenHeight*greenNColors)
                if len(greenFrameRaw) != greenWidth*greenHeight*greenNColors:
                    break # we are done
                for pix in range(greenWidth*greenHeight*greenNColors):
                    greenFrame[pix] = greenFrameRaw[pix]
                greenFrame = np.reshape(greenFrame, (greenHeight,greenWidth,greenNColors))
                
                backgroundFrame = np.zeros([backgroundWidth*backgroundHeight*backgroundNColors,1]).astype(np.uint8)
                backgroundFrameRaw = backgroundVideoStream.read(backgroundWidth*backgroundHeight*backgroundNColors)
                if len(backgroundFrameRaw) != backgroundWidth*backgroundHeight*backgroundNColors:
                    break # we are done
                for pix in range(backgroundWidth*backgroundHeight*backgroundNColors):
                   backgroundFrame[pix] = backgroundFrameRaw[pix]
                backgroundFrame = np.reshape(backgroundFrame, (backgroundHeight,backgroundWidth,backgroundNColors))

            if not success_g or not success_b:
                break
            
            # Rotate the background image if so desired
            if bgRotate_deg != 0:
                backgroundFrame = np.rot90(backgroundFrame, k=int(bgRotate_deg/90), axes=(0,1))
            
            # Inject the background
            frame, mask = gst.injectBackground(greenFrame, backgroundFrame,[600,300])
            
            # Write mask to output video stream
            # .avi format
            mask3 = np.repeat(mask.astype('uint8')[:,:,np.newaxis]*255,3,axis=2)
            outMaskStream.write(mask3) 
            # .jpg sequence
            mname = "mask_%04d.jpg" % iFrame
            if not os.path.isdir(os.path.join("sequences",outVideoBase)):
                os.mkdir(os.path.join("sequences",outVideoBase))
            cv2.imwrite(os.path.join("sequences",outVideoBase,mname), mask3) 
            
            # Write frame to output video stream
            # .avi format
            outStream.write(frame)
            fname = "frame_%04d.jpg" % iFrame
            # .jpg sequence
            if not os.path.isdir(os.path.join("sequences",outVideoBase)):
                os.mkdir(os.path.join("sequences",outVideoBase))
            cv2.imwrite(os.path.join("sequences",outVideoBase,fname), frame)
            
            print('Injecting onto frame ' + str(iFrame))
            iFrame += 1
        # while success
        
    finally:       
        # Save it off so we can play the video
        print("Exiting cleanly")
        outStream.release()
        outMaskStream.release()
        print("Output video file: "+outVideoFile) 
        print("Output mask file: "+outMaskFile)
        print("Output image folder: " + os.path.join("sequences",outVideoBase)) 
    
# end run_videoInjection()

# Run with defaults if at highest level
if __name__ == "__main__":
    #run_videoInjection("defaultGreenscreenVideo.avi", "defaultBackgroundVideo.avi")
    #run_videoInjection("defaultGreenscreenVideo.vraw", defaultBackgroundVideo.vraw")
    #run_videoInjection("defaultGreenscreenVideo.avi", os.path.join("backgroundVideos","sophieTricycle.avi"),270)
    #run_videoInjection("defaultGreenscreenVideo.avi", os.path.join("backgroundVideos","EPC_ramp.avi"),270)
    #run_videoInjection("defaultGreenscreenVideo.avi", os.path.join("backgroundVideos","EPC_hallway.avi"),270)
    #run_videoInjection("defaultGreenscreenVideo.avi", os.path.join("backgroundVideos","PHO_hallway.avi"),270)
    run_videoInjection("defaultGreenscreenVideo.avi", os.path.join("backgroundVideos","BOS_trainSidewalk.avi"),270)
    
