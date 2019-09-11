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

def findGreenscreen(greenImage, screenColor=[0,1,0], colorVariance=0.1):
    # Check that the image is indeed a 3-D array ranged 0 to 1
    nx,ny,nc = greenImage.shape
    
    # Check validity of screenColor
    assert len(screenColor) == 3
    assert np.fabs(np.linalg.norm(screenColor) - 1.0) < 0.001
    
    # Check validity of colorVariance
    assert colorVariance >= 0
    assert colorVariance <= 1
    
    # Normalize each image pixel?
    
    # Dot each pixel by screenColor
    greenScore = np.dot(greenImage.reshape(nx*ny,nc), screenColor)
    greenScore = np.reshape(greenScore,[nx,ny,nc])
    
    # Apply a threshold to the green score to create the mask
    greenMask = greenScore > (1.0 - colorVariance)
    
    return greenMask
    