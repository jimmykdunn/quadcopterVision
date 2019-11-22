# quadcopterVision
# MS ECE Thesis Project code
# James Dunn, Boston University, 2019-2020

Code for visual detection of quadcopters from video feeds for use in formation control.
Languages: Python with tensorflow

ABSTRACT
The ability to command a team of quadrotor drones to maintain a formation is an important capability for applications such as area surveillance, search and rescue, agriculture, and reconnaissance.  Of particular interest is operation in environments where radio and/or GPS may be denied or not sufficiently accurate for the desired application.  This is common indoors, in military scenarios due to electronic interference (jamming), and in unmapped or dynamic areas where reliance on existing maps for absolute geolocation is insufficient for relative localization of the quadcopters.  Scenarios where one or more of the quadrotors is non-cooperative, such as a seek-and-destroy type of mission, are also possible when using only visual sensors.

To address these issues, we develop and implement an algorithm that discriminates between quadrotor pixels and non-quadrotor pixels.  We implement the algorithm on a single-board ODROiD XU4 mounted to a quadrotor drone. An standard visual-band webcam is the only sensor used.

The core of our detection and localization algorithm is an hourglass convolutional neural network (CNN) that generates heatmaps - pixel-by-pixel likelihood estimates of the presence of a quadrotor.  A version of the Siamese networks technique is added for clutter mitigation and enforcement of temporal smoothness. Specifically we enforce the similarity of temporally adjacent frames with a Siamese loss term. The resulting centroids of each quadcopter segmentation are Kalman-filtered in time as well.

Total latency - specifically the time from frame capture to reporting the localization estimate of the observed quadcopter(s) - is kept at approximately 20ms when implmented on the ODROiD XU4. 
