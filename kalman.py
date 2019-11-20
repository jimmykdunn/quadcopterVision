# -*- coding: utf-8 -*-
"""
FILE: kalman.py
DESCRIPTION:
    Defines kalman_filter class for initiating and maintaining a 2-element
    state through time.

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: November 2019
    
Based on kalman.py from ME740 project in Spring 2019.
"""
import numpy as np

# Holds the kalman filter state for the target
# State is a 4-element vector (x, y, vx, vy), plus a 4x4 
# covariance matrix (sigx, sigY, sigVX, sigVY) diagonal terms.
# We initalize them with values upon construction of the class.
class kalman_filter:
    def __init__(self, x, y, vx, vy, sigX, sigY, sigVX, sigVY):
        
        self.stateVector = [x, y, vx, vy]
        
        self.covMatrix = [[sigX, 0,    0,     0    ], \
                          [0,    sigY, 0,     0    ], \
                          [0,    0,    sigVX, 0    ], \
                          [0,    0,    0,     sigVY]]
                          
        # Process noise matrix. In our setup, this captures our estimate
        # of the uncertainty related to the motion of the target.
        # In reality, this is a function of the time since the last measurement,
        # but since we have a somewhat stable framerate, we let it be constant.
        self.Q = [[0.2,    0,    0,    0], \
                  [0,    0.2,    0,    0], \
                  [0,      0, 0.05,    0], \
                  [0,      0,    0, 0.05]]
        #self.Q = [[0, 0, 0, 0], \
        #          [0, 0, 0, 0], \
        #          [0, 0, 0, 0], \
        #          [0, 0, 0, 0]]
                  
        # State to sensor reading mapping matrix 
        # We measure x and y directly, and don't measure velocity directly, so 
        # this is a 2x4 matrix as follows     
        self.H = [[1, 0, 0, 0], \
                  [0, 1, 0, 0]]
	

    # Takes the state of the target and projects it forward by dt seconds.
    # This should include an induced motion control per the applied controls,
    # but we have not gotten that far yet.
    def project(self, dt):
        updateMatrix = [[1, 0, dt, 0], \
                        [0, 1, 0, dt], \
                        [0, 0, 1,  0], \
                        [0, 0, 0,  1]]
                     
        applied_dx = 0.0
        applied_dy = 0.0
        appliedControl = [applied_dx, applied_dy, 0, 0]
        
        self.stateVector = np.dot(updateMatrix, self.stateVector) + appliedControl
        self.covMatrix = np.dot(np.dot(updateMatrix, self.covMatrix), np.transpose(updateMatrix)) + self.Q
        

    # Updates the state using the measurement. Note that it is assumed we have
    # already projected the state forward at this point, so the update step is
    # literally just changing the state per the measurement.
    # K = P H' (H P H' + R)^-1
    # x = x + K (z - H x)
    # P = P - K H P
    # measurement is a 2x1 vector [x, y]
    # msmtErrorMatrix is a 2x2 matrix [sigX, 0   ]
    #                                 [0,    sigY]
    def update(self, measurement, msmtErrorMatrix):
        # Create the Kalman gain matrix, which carries the relationship between
        # the error of the measurements and the existing covaraince matrix.
        # Only need the measurement error to calculate it.
    	K = np.dot(np.dot(self.covMatrix, np.transpose(self.H)), \
    		      np.linalg.inv(np.dot(np.dot(self.H, self.covMatrix), np.transpose(self.H)) \
    		             + msmtErrorMatrix) \
    		  )
  
        # Update the state vector. Simply the state we projected to plus the
        # Kalman-gain-weighted difference between the measurement and the 
        # projected state vector.
    	self.stateVector = self.stateVector + \
                           np.dot(K, measurement - np.dot(self.H, self.stateVector))
    
        # Update the covariance matrix based on the kalman gain
    	self.covMatrix = self.covMatrix - np.dot(K, np.dot(self.H, self.covMatrix))
    	
    # Prints the state of the Kalman filter
    def printState(self):
    	print(self.stateVector)
    	
    # Prints the covariance of the Kalman filter
    def printCovariance(self):
        print(self.covMatrix)
