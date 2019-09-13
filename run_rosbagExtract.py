# -*- coding: utf-8 -*-
"""
FUNCTION: run_rosbagExtract
DESCRIPTION:
    Extracts video data from a rosbag and writes to binary video file or avi.
    
INPUTS: 
    bagfile: path to a rosbag file with embedded video
    
OUTPUTS: 
    Writes video embedded in rosbag to file.
    Video file will have the same name as the rosbag file but with ".avi" or 
    ".vraw" extension.

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""

import numpy as np
import struct
import os
import cv2

SAVE_AS_AVI = True # True to save in avi format (small & fast but low quality)

def run_rosbagExtract(bagfile):
    
    # Open up the bag
    bagptr = open(bagfile, "rb")
    iImage = 0
    
    # Start crawling
    byte = 0
    bytesRead = 0
    
    # Read the version
    version = ''
    while byte != b'\n':
        byte = bagptr.read(1)
        bytesRead += 1
        version += byte.decode('utf-8')
    print(version)
    
    try:
    
        # The rest of the file is a series of records whose format is
        # <header_len> (4 bytes)
        # <header> (header_len bytes)
        # <data_len> (4 bytes)
        # <data> (data_len bytes)
        # This allows easy random access
        recordCount = 0
        videoInitialized = False
        videoStream = 0
        
        # Loop over the records in the rosbag
        while True:
            # Read record header
            header_len = struct.unpack('i', bagptr.read(4))[0]
            bytesRead += 4
            header = bagptr.read(header_len)
            bytesRead += header_len
            
            # Read record data
            data_len = struct.unpack('i', bagptr.read(4))[0]
            bytesRead += 4
            data = bagptr.read(data_len)
            bytesRead += data_len
    
            # Search the data in this record for imagery
            ick = 0
            while ick < len(data):
                # First record will contain metadata
                if data[ick:].find(b'type=sensor_msgs/Image') != -1:
                    imageIdx = data[ick:].find(b'type=sensor_msgs/Image')
                    
                    # Parse relevant metadata
                    starti = ick+imageIdx # move up to start of image block
                    timeSec = struct.unpack('ll', data[starti+56:starti+64])[0]
                    ny = struct.unpack('i', data[starti+84:starti+88])[0]
                    nx = struct.unpack('i', data[starti+88:starti+92])[0]
                    bpp=1
                    
                    # Initilaize the output video stream if we have not yet
                    if not videoInitialized:
                        if SAVE_AS_AVI:
                            # Get the video stream name ready
                            framerate = 30.0 # fps
                            videoFilename = os.path.splitext(bagfile)[0] + '.avi'
                            videoStream = cv2.VideoWriter(videoFilename,cv2.VideoWriter_fourcc('X','V','I','D'), \
                                                        framerate, (nx, ny))
                        else:
                            # Save binary raw bgr values
                            videoFilename = os.path.splitext(bagfile)[0] + '.vraw'
                            videoStream = open(videoFilename, "wb")
                            nc = 3
                            videoStream.write(nx.to_bytes(4,"little"))
                            videoStream.write(ny.to_bytes(4,"little"))
                            videoStream.write(nc.to_bytes(4,"little"))
                            
                        videoInitialized= True
                
                # Remaining records do not contain metadata. Find with 'bgr8' flag
                if data[ick:].find(b'bgr8') != -1:
                    imageIdx = data[ick:].find(b'bgr8')
                    starti = ick+imageIdx # location of 'bgr8'
                    timeSec = struct.unpack('ll', data[starti-50:starti-42])[0]
                    stepSz = struct.unpack('i', data[starti+5:starti+9])[0] # full row length in bytes
                    imgNBytes = struct.unpack('i', data[starti+9:starti+13])[0]
                    imgNPix = imgNBytes * bpp
                    
                    
                    # Actual image matter extraction and conversion
                    img0 = starti+13 # first byte index of image data
                    imgBytes = data[img0:img0+imgNBytes]
                    
                    # Print info about image
                    print("IMAGE " + str(iImage))
                    print("    Bytepos = " + str(bytesRead))
                    print("    Time (s) = " + str(timeSec))
                    print("    (nx,ny) = ("+str(nx)+","+str(ny)+")")
                    print("    stepSz = " + str(stepSz))
                    iImage += 1
                    
                    # Extract image from data buffer
                    imageFlat = np.zeros([imgNPix,1]).astype('uint8')
                    for ip in range(imgNPix):
                        imageFlat[ip]=ord(imgBytes[bpp*ip:bpp+bpp*ip])
                    image = np.reshape(imageFlat,[ny,nx,3])  
                                        
                    # Save image to video stream
                    videoStream.write(image)
                    ick = img0+imgNBytes
                
                # if found image start flag
                else:
                    # No more images found in this record, break
                    break
            # while still have data left in this record
            
            recordCount += 1
        
    except:
        print("End of file (or error)")
            
    finally:
        bagptr.close()
        
        if SAVE_AS_AVI:
            # Save it off so we can play the video
            print("Releasing video stream handle")
            videoStream.release()
        else:
            videoStream.close()

# end run_rosbagExtract()

# Run with defaults if at highest level
if __name__ == "__main__":
    run_rosbagExtract("C:\\Users\\jimmy\\OneDrive\\Documents\\gradSchool\\thesis\\data\\rosbags\\helloworld.bag")
    #run_rosbagExtract("C:\\Users\\jimmy\\OneDrive\\Documents\\gradSchool\\thesis\\data\\rosbags\\greenTest_01.bag")
    