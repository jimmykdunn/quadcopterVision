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

def rosbag_crawler(bagfile):
    # Open up the bag
    bagptr = open(bagfile, "rb")
    
    # Start crawling
    try:
        byte = 0
        
        # Read the version
        version = ''
        while byte != b'\n':
            byte = bagptr.read(1)
            version += byte.decode('utf-8')
        print(version)
        
        # The rest of the file is a series of records whose format is
        # <header_len> (4 bytes)
        # <header> (header_len bytes)
        # <data_len> (4 bytes)
        # <data> (data_len bytes)
        # This allows easy random access
        recordCount = 0
        #while True:
        for i in range(3):
            header_len = struct.unpack('i', bagptr.read(4))[0]
            header = bagptr.read(header_len)
            isChunk = parse_header(header)
            
            data_len = struct.unpack('i', bagptr.read(4))[0]
            print('DATA')
            print('    data length=' + str(data_len))
            data = bagptr.read(data_len)
            
            if isChunk:
                # Interesting stuff here. Figure out how to parse it
                bf = 1
                
                #/quad_bb93/face_detection/image appears to prepend imagery
                # and it occurs in the middle of data chunks. Search for it!
                
                # Additionally, there is a bunch of text describing it under
                # "message_definition"
                pass
                
            recordCount += 1
            print('-------------------')
    except:
        print("End of file probably")
            
            
    #    while byte != "":
    #        byte = bagptr.read(1)
    #        print(byte)
    finally:
        bagptr.close()
    
    
# end rosbag_crawler()
    
# Parses and prints a header    
def parse_header(header):
    isChunk = False
    i = 0
    print('HEADER')
    while i<len(header):
        # Read field length
        fieldLen = struct.unpack('i', header[i:i+4])[0]
        i += 4
        
        # Read field name
        iFieldStart = i
        ifield = 0
        fieldName = ''
        while chr(header[i]) != '=':
            fieldName += chr(header[i])
            i += 1
            ifield += 1
        i += 1    
        
        # Read field value
        fieldSize = iFieldStart + fieldLen - i
        
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
            
        print('    '+fieldName + '=' + str(value))
        i += fieldSize
        
    return isChunk

# Run with defaults if at highest level
if __name__ == "__main__":
    rosbag_crawler("C:\\Users\\jimmy\\OneDrive\\Documents\\gradSchool\\thesis\\data\\rosbags\\greenTest_01.bag")
    