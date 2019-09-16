# quadcopterVision
# MS ECE Thesis Project code
# James Dunn, Boston University, 2019-2020

Code for visual detection of quadcopters from video feeds for use in formation control.
Languages: Python, c++, tensorFlow


#TO DO LIST

-Varying levels of zoom for faking different resolutions/distances
-Varying levels of lighting (lights on and off?)
-Varying camera angles AND drone angles
-Multiple levels of resolution for input to the CNN? Think localization with low res and then classification with high res. Great possibility for main thrust of thesis!
-Short temporal stack of frames for better detection based on propellor movements
-Remember to include images with AND without drones in them!
-Save off the greenscreen mask for use as truth! Can visualize it in videos as a green outline of the truth.

A potential place for uniqueness is the use of short temporal stacks as CNN input to let motion be learned.
