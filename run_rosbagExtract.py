# -*- coding: utf-8 -*-
"""
FUNCTION: run_rosbagExtract
DESCRIPTION:
    Extracts relevant information from a recorded rosbag, including video data.
    
INPUTS: 
    rosbagFile: location of a rosbag file
    
OUTPUTS: 
    Writes relevant videos to file

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
import cv2
import rosbagpy

def run_rosbagExtract():
    # Open up the bag
    
    # Write videos to standard file
    
    pass
    
# end run_rosbagExtract()

# Run with defaults if at highest level
if __name__ == "__main__":
    run_rosbagExtract("C:\\Users\\jimmy\\OneDrive\\Documents\\gradSchool\\thesis\\data\\rosbags\\greenTest_01.bag")
    