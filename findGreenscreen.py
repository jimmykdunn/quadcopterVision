# -*- coding: utf-8 -*-
"""
FUNCTION: findGreenscreen
DESCRIPTION:
    Takes an image with a greenscreen background and segregates greenscreen 
    pixels from foreground pixels.
INPUTS: 
    greenImage: 2D image with greenscreen background
    screenColor: [R,G,B]-triplet (range 0 to 1) of expected screen color
    colorVariance: fraction (0 to 1) setting allowed variance about screenColor
    
RETURNS: 
    greenMask: boolean mask - true at greenscreen pixels, false elsewhere

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
import numpy as np
import cv2

def findGreenscreen(greenImageIn, screenColorHSV=[41,63,138], colorVariance=30.0):
    # Check that the image is indeed a 3-D array
    nx,ny,nc = greenImageIn.shape
    
    # Check validity of screenColor
    assert len(screenColorHSV) == 3    
    
    # Convert to HSV colorspace
    greenImage_hsv = cv2.cvtColor(greenImageIn, cv2.COLOR_BGR2HSV)
    
    #greenScore = np.mean(np.fabs(greenImage-screenColor),axis=2)
    greenScore = np.fabs(greenImage_hsv - screenColorHSV).astype('float')
    greenScore *= [3, 0.510, 0.934] # weight #cv30
    greenScore = np.mean(greenScore,axis=2)
    
    # Apply a threshold to the green score to create the mask
    greenMask = greenScore < colorVariance
    
    return greenMask
    