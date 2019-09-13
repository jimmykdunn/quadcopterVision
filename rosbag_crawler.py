# -*- coding: utf-8 -*-
"""
FUNCTION: rosbag_crawler
DESCRIPTION:
    Extracts relevant information from a recorded rosbag, including video data.
    
INPUTS: 
    rosbagFile: location of a rosbag file
    
OUTPUTS: 
    Extracts important data from rosbag file

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
import os
import cv2

def rosbag_crawler(bagfile):
    # Open up the bag
    bagptr = open(bagfile, "rb")
    iImage = 0
    
    # Start crawling
    #try:
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
        while True:
        #for i in range(1000):
            header_len = struct.unpack('i', bagptr.read(4))[0]
            bytesRead += 4
            header = bagptr.read(header_len)
            bytesRead += header_len
            isChunk = parse_header(header)
            
            data_len = struct.unpack('i', bagptr.read(4))[0]
            bytesRead += 4
            #print('DATA')
            #print('    data length=' + str(data_len))
            data = bagptr.read(data_len)
            bytesRead += data_len
            
            #print("Bytes Read: " + str(bytesRead))
            
            if isChunk:
                ick = 0
                while ick < len(data):
                    if data[ick:].find(b'type=sensor_msgs/Image') != -1:
                        imageIdx = data[ick:].find(b'type=sensor_msgs/Image')
                        
                        # Print identifying information for debugging
                        #print(str(data[ick+imageIdx-3000:ick+imageIdx]))
                        
                        # Parse all the metadata
                        starti = ick+imageIdx # move up to start of image block
                        foo1 = struct.unpack('i', data[starti+22:starti+26])[0]
                        foo2 = struct.unpack('i', data[starti+26:starti+30])[0]
                        connStr = data[starti+30:starti+35]
                        conn1 = struct.unpack('i', data[starti+35:starti+39])[0]
                        conn2 = struct.unpack('i', data[starti+39:starti+43])[0]
                        opStr = data[starti+43:starti+46]
                        op = ord(data[starti+46:starti+47])
                        timestr = data[starti+51:starti+56]
                        timeSec = struct.unpack('ll', data[starti+56:starti+64])[0]
                        timeNsec = struct.unpack('ll', data[starti+64:starti+72])[0]
                        ny = struct.unpack('i', data[starti+84:starti+88])[0]
                        nx = struct.unpack('i', data[starti+88:starti+92])[0]
                        nch = struct.unpack('i', data[starti+92:starti+96])[0]
                        bpp=1
                        
                        if not videoInitialized:
                            # Get the video stream name ready
                            framerate = 30.0 # fps
                            videoFilename = os.path.splitext(bagfile)[0] + '.avi'
                            #videoStream = cv2.VideoWriter(videoFilename,cv2.VideoWriter_fourcc(*'DIVX'), \
                            #                            framerate, (nx, ny))
                            videoStream = cv2.VideoWriter(videoFilename,cv2.VideoWriter_fourcc('X','V','I','D'), \
                                                        framerate, (nx, ny))
                            #videoStream = cv2.VideoWriter(videoFilename,cv2.VideoWriter_fourcc(*'MJPG'), \
                            #                             framerate, (nx, ny))
                            videoInitialized= True
                    
                    if data[ick:].find(b'bgr8') != -1:
                        imageIdx = data[ick:].find(b'bgr8')
                        starti = ick+imageIdx # location of 'bgr8'
                        timestr = data[starti-45:starti-50]
                        timeSec = struct.unpack('ll', data[starti-50:starti-42])[0]
                        #timeNsec = struct.unpack('ll', data[starti-42:starti-34])[0]
                        encoding = str(data[starti+0:starti+4]) # bgr8
                        bigEndian = data[starti+4]
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
                        #print("    Time (ns) = " + str(timeNsec))
                        print("    (nx,ny) = ("+str(nx)+","+str(ny)+")")
                        print("    stepSz = " + str(stepSz))
                        iImage += 1
                        
                        imageFlat = np.zeros([imgNPix,1]).astype('uint8')
                        for ip in range(imgNPix):
                            imageFlat[ip]=ord(imgBytes[bpp*ip:bpp+bpp*ip])
                        image = np.reshape(imageFlat,[ny,nx,3])  
                        #plt.figure()
                        #plt.imshow(image)
                        
                        
                        # Save image to file
                        videoStream.write(image)
                        #imNumStr = "%04d" % iImage
                        #scipy.misc.imsave('images\\image_'+imNumStr+'.jpg', image)
                        
                        
                        # Arbitrary stop point
                        #if iImage >= 100:
                        #    print("Stopping here voluntarily")
                        #    return
                        
                        ick = img0+imgNBytes
                    
                    # if found image start flag
                    else:
                        # No more images found in this record, break
                        break
                # while still have data left in this record
                
            # if is a chunk
            recordCount += 1
        
    except:
        print("End of file probably")
            
            
    #    while byte != "":
    #        byte = bagptr.read(1)
    #        print(byte)
    finally:
        bagptr.close()
        
        # Save it off so we can play the video
        print("Releasing video stream handle")
        videoStream.release()
    
    
# end rosbag_crawler()
    
# Parses and prints a header    
def parse_header(header,spaces=''):
    isChunk = False
    i = 0
    #print(spaces+'HEADER: ' + str(len(header)) + " bytes")
    while i<len(header):
        # Read field length
        fieldLen = struct.unpack('i', header[i:i+4])[0]
        i += 4
        
        # Read field name
        iFieldStart = i
        ifield = 0
        fieldName = ''
        while chr(header[i]) != '=' and ifield < fieldLen:
            fieldName += chr(header[i])
            i += 1
            ifield += 1
            if i >= len(header):
                break
        i += 1    
        
        # Read field value
        fieldSize = iFieldStart + fieldLen - i
        printField = True
        
        if fieldName == 'op':
            if header[i:i+fieldSize] == b'\x02':
                value = 'message data'
            elif header[i:i+fieldSize] == b'\x03':
                value = 'bag header'
            elif header[i:i+fieldSize] == b'\x04':
                value = 'index data'
            elif header[i:i+fieldSize] == b'\x05':
                value = 'chunk'
                isChunk = True
            elif header[i:i+fieldSize] == b'\x06':
                value = 'chunk info'
            elif header[i:i+fieldSize] == b'\x07':
                value = 'connection'
        elif fieldName == 'compression':
            value = str(header[i:i+fieldSize])
        elif fieldSize == 4 or fieldSize == 8:
            if fieldSize == 4:
                fieldType = 'i' # 4-byte int
            if fieldSize == 8:
                if 'time' in fieldName:
                    fieldType = 'q' # double
                else:
                    fieldType = 'q' # 8-byte int
        
            value = struct.unpack(fieldType, header[i:i+fieldSize])[0]
        else:
            value = header[i:i+fieldSize]
            printField = False
            
        if printField:
            #print(spaces+'    '+fieldName + '=' + str(value))
            pass
        else:
            #print(spaces+'    Field size: '+ str(fieldSize))
            pass
        if fieldSize < 0:
            stophere = 0
        else:
            i += fieldSize
        
    return isChunk

# Run with defaults if at highest level
if __name__ == "__main__":
    #rosbag_crawler("C:\\Users\\jimmy\\OneDrive\\Documents\\gradSchool\\thesis\\data\\rosbags\\helloworld.bag")
    rosbag_crawler("C:\\Users\\jimmy\\OneDrive\\Documents\\gradSchool\\thesis\\data\\rosbags\\greenTest_01.bag")
    