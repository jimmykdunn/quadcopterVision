# -*- coding: utf-8 -*-
"""
FILE: importData.py
DESCRIPTION:
    Data import function

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: November 2019
"""
import os
import videoUtilities as vu
import numpy as np

"""
importRoboticsLabData()
    Imports all robotics lab data
INPUTS:
    None
OPTIONAL INPUTS:
    quickTest: Set to true to just import one of the videos for quick debugging
    tests. Default False.
RETURNS:
    x_all: all input images with injected quadrotors (including augmentations)
    y_all: all associated boolean truth masks
    id_all: String identifying the frame and augmentation of each input x_all.
            "****_%%%%", where **** is the temporal frame number and %%%% is the
            augmentation index.
    id_all_plus: id_all with an additional "video" index appended, i.e.
            "****_%%%%_$$" where **** and %%%% are as above, and $$ represents
            the index of the video that each frame came from.
"""
def importRoboticsLabData(quickTest=False):
    
    x_set1, y_set1, id_set1 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_mirror","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_mirror","augMask_"))
    id_set1_plus = [id+"_01" for id in id_set1]
    x_set2, y_set2, id_set2 = vu.pull_aug_sequence(
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_mirror","augImage_"),
        os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_mirror","augMask_"))
    id_set2_plus = [id+"_02" for id in id_set2]
    x_all = np.concatenate([x_set1,x_set2],axis=0)
    y_all = np.concatenate([y_set1,y_set2],axis=0)
    id_all = np.concatenate([id_set1,id_set2],axis=0)
    id_all_plus = np.concatenate([id_set1_plus,id_set2_plus],axis=0)
    
        
    if not quickTest:
        x_set3, y_set3, id_set3 = vu.pull_aug_sequence(
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_mirror","augImage_"),
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_mirror","augMask_"))
        id_set3_plus = [id+"_03" for id in id_set3]
        x_set4, y_set4, id_set4 = vu.pull_aug_sequence(
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_mirror","augImage_"),
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_mirror","augMask_"))
        id_set4_plus = [id+"_04" for id in id_set4]
        x_set5, y_set5, id_set5 = vu.pull_aug_sequence(
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_baby_mirror","augImage_"),
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab1_64x48_baby_mirror","augMask_"))
        id_set5_plus = [id+"_05" for id in id_set5]
        x_set6, y_set6, id_set6 = vu.pull_aug_sequence(
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_baby_mirror","augImage_"),
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab2_64x48_baby_mirror","augMask_"))
        id_set6_plus = [id+"_06" for id in id_set6]
        x_set7, y_set7, id_set7 = vu.pull_aug_sequence(
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_baby_mirror","augImage_"),
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab3_64x48_baby_mirror","augMask_"))
        id_set7_plus = [id+"_07" for id in id_set7]
        x_set8, y_set8, id_set8 = vu.pull_aug_sequence(
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_baby_mirror","augImage_"),
            os.path.join("augmentedContinuousSequences","defaultGreenscreenVideo_over_roboticsLab4_64x48_baby_mirror","augMask_"))
        id_set8_plus = [id+"_08" for id in id_set8]
    
        x_all = np.concatenate([x_set1,x_set2,x_set3,x_set4,x_set5,x_set6,x_set7,x_set8],axis=0)
        y_all = np.concatenate([y_set1,y_set2,y_set3,y_set4,y_set5,y_set6,y_set7,y_set8],axis=0)
        id_all = np.concatenate([id_set1,id_set2,id_set3,id_set4,id_set5,id_set6,id_set7,id_set8],axis=0)
        id_all_plus = np.concatenate([id_set1_plus,id_set2_plus,id_set3_plus,id_set4_plus,id_set5_plus,id_set6_plus,id_set7_plus,id_set8_plus],axis=0)
    # end if not quickTest
    
    return x_all, y_all, id_all, id_all_plus
    
# end importRoboticsLabData()    
