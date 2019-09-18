# -*- coding: utf-8 -*-
"""
FILE: greenscreenTools.py
Contains functions useful for working with greenscreen images and videos.

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: September 2019
"""

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
"""
import copy
import numpy as np
import cv2

def injectBackground(greenImage, background, shift=[0,0]):
    
    # Find background pixels with findGreenscreen.py. Use default parameters.
    greenMask = findGreenscreen(greenImage)
    
    # Change the size of the greenscreened frame (for example, by 
    # zooming in or simply zeropadding) so that it matches the size of 
    # the background video.
    if greenImage.shape != background.shape:
        greenImage, greenMask = smartSizeMatch(greenImage, greenMask, background.shape)
        
    # Optionally shift the greenImage (target) over the background image
    if len(shift) == 2:
        shift.append(0) # tack on color shift (none)
    greenImage = np.roll(greenImage,shift[0],axis=0)
    greenImage = np.roll(greenImage,shift[1],axis=1)
    greenMask = np.roll(greenMask,shift[0],axis=0)
    greenMask = np.roll(greenMask,shift[1],axis=1)
    
    # Replace greenscreen pixels with background pixels
    image = copy.deepcopy(greenImage)
    image[greenMask] = background[greenMask]
    
    return image, greenMask

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
"""

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


"""
FUNCTION: smartSizeMatch
DESCRIPTION:
    Takes an image and resizes it to the desired size via zeropadding the
    edges (default), and optionally shifting or scaling (not yet implemented)
INPUTS: 
    image: 2D image (with color channel)
    mask: binary mask with true (1) for pixels that are part of the greenscreen
    newShape: desired size of image as a [nx,ny,nc] triplet 
    
RETURNS: 
    newImage: resized image
    newGreenMask: resized mask
"""
def smartSizeMatch(image, greenMask, newShape):
    nx,ny,nc = image.shape
    
    # First attempt: just zeropad the image and truepad the mask
    newImage = np.zeros(newShape).astype(np.uint8)
    newMask = np.zeros(newShape[:2]) == 0
    newImage[0:nx,0:ny,0:nc] = image
    newMask[0:nx,0:ny] = greenMask

    return newImage, newMask
