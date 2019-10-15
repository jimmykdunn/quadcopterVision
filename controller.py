# -*- coding: utf-8 -*-
"""
FILE: controller
DESCRIPTION:
    Runs the UAV camera, ingests and processes images, displays a live video
    feed, and calculates feedback controls. 

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""

import webcam_utils as wcam

def run():
    # Initialize the video stream from the camera
    wcam.videoStream()

# Run if called directly
if __name__ == "__main__":
    run()
