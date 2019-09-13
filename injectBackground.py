# -*- coding: utf-8 -*-
"""
FUNCTION: injectBackground
DESCRIPTION:
    Takes an image with a greenscreen background and replaces the green pixels
    with background image pixels.
INPUTS: 
    greenImage: 2D image with greenscreen background
    background: 2D image to go over greenscreen (same size as greenImage)
    
RETURNS: 
    image: image with greenscreen pixels replaced with background pixels

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""
import copy
import findGreenscreen

def injectBackground(greenImage, background):
    # Check that greenImage and background are the same shape
    assert greenImage.shape == background.shape
    
    # Find background pixels with findGreenscreen.py. Use default parameters.
    greenMask = findGreenscreen.findGreenscreen(greenImage)
    
    # Replace greenscreen pixels with background pixels
    image = copy.deepcopy(greenImage)
    image[greenMask] = background[greenMask]
    
    return image