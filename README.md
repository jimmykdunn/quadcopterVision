quadcopterVision
MS ECE Thesis Project code
James Dunn, Boston University, 2019-2020

Code for visual detection of quadcopters from video feeds for use in formation control.
Languages: Python with tensorflow

==================================

DEPENDENCIES:

cv2 (version 4.0.1 recommended, use other versions at your own risk)

tensorflow (version 1.13.1 recommended, use other versions at your own risk)

Most recent versions of : numpy, os, random, copy, matplotlib, datetime, shutil, time, imutils, sys, struct

==================================

DIRECTIONS:

The following directions detail how to get data ready, augment and prepare it for the neural network, train the neural network, and run forward passes of the neural network. These are general directions.  You will probably need to look up some specifics in the code files themselves to run things exactly how you want.

1. Get some background videos in .avi format that represent the typical environment that you will use the quadcopter in. These are NOT provided on the git repo due to size restrictions.

2. Use run_videoInjection.py with the .avi files from your background videos and defaultGreenscreenVideo.avi.  Take care to set scaling parametrs appropriately for the size of the injected quadcopter over your background. This will create a new .avi with the injected quadcopter, as well as a matching sequence of jpg images and injection masks in the "sequences" directory.

3. Run augment_continuous_sequence() in videoUtilities.py.  videoUtilities.py is set up to do this when called from the command line ("python videoUtilities.py").  This will take the injected videos that you just saved into the sequences directory and apply a series of augmentations to them, as well as crop them down to the proper size for neural network input. This can take a little while to run. Take care to set the first and last frames to augment so that you are only augmenting those frames which you actually want to use to train the neural network.  The result will be a series of augmented images and corresponding masks in the augmentedContinuousSequences directory. The numbers appended to the files are "frameNumber_augmentationIndex" in a "####_%%%%" format.

4. Run the importFoldSave(4,"folds") function in nfold_siamese_hourglass_cnn.py. This will partition the augmented sequences you just created into multiple (nominally 4) folds so that we can run N-fold cross validation with our neural network training. The folded data will be saved to the "folds" directory for later use.  Note that the "importRoboticsLabData()" function in importData.py may need to be adjusted depending on what data you actually want to use.

5. Run "python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 someNameForYourTrainedNetwork".  0.1 represents the siamese loss term's weight - 0.1 is the recommended value.  the two 0.00 0.00 that follow are the weights for other loss terms that are not recommended to be used.  Just leave them as 0.00 unless you really know what you're doing.  Hou can change the number of epochs (nEpochs) and batch size (batchSize) as you like.  This will run the nfold training for each fold SERIALLY. The trained network will be checkpointed every 1000 epochs, and will be converted into protobuf (.pb) format at the end and saved to "./savedNetworks/someNameForYourTrainedNetwork_fold#".

5. The above step can take quite literally DAYS to run, so the recommended way to do it is to submit one job per fold your queue using "qsub submit_run_hourglass_cnn.sh" with the "python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 someNameForYourTrainedNetwork 0" line in submit_run_hourglass_cnn.sh set as you like.  The final "0" argument to nfold_siamese_hourglass_cnn.py represents the fold index to train.  Set it to 0, then do a "qsub submit_run_hourglass_cnn.sh". Then set it to 1 and do another "qsub submit_run_hourglass_cnn.sh". Continue until you have submitted one job for each fold you are running.

6. Once you have run 1000 or more epochs, you will get .ckpt files in your "savedNetworks/someNameForYourTrainedNetwork_fold#" directory. You can set the directories at the bottom of use_hourglass_cnn.py and then run use_hourglass_cnn.py to read the appropriate .ckpt file and run the partially-trained network on some example images to check in on the status of your network training midway through.

7. At this point, you should have one trained network per fold in your "savedNetworks" folder.  What we will use going forward is the ".pb" files in your savedNetworks/someNameForYourTrainedNetwork_fold# directories.  Note that each of these networks has been trained with the data in the OTHER N-1 folds so that you can properly run cross-validation using the data from your fold-of-interest while maintaining train/test separation.

8. You can now run a proper N-fold analysis on your trained networks using analyzePerformance.py.  Run "runNFoldPerformanceAnalysis(4, 'folds', os.path.join('savedNetworks','someNameForYourTrainedNetwork'), modelName = "modelFinal_full") from python to read in all the data with readFoldedImages() (you may need to tweak that function to use your data paths). This will read in all the data, run it on the appropriate "fold" trained network, and print out resulting confusion matrices for a series of heatmap thresholds. It is recommended that you direct the text output from runNFoldPerformanceAnalysis() into a log file so that you can use it later.

9. If desired, you can compare the results of different networks with the code at the end of analyzePerformance.py using the rocCurveComparison() function on the logs that you captured from the previous step.  This will generate a plot of ROC curves for all of the log files that you feed to it.

10. You can also run your trained network live on video streamed from a camera with controller.py. For my thesis work, this was done onboard an ODROiD XU4 mounted to a quadcopter, but the code should work with any camera you have connected to whatever computer you are working with.  You can also run on already-captured data by using the "filestream" argument to the run() function in controller.py. Check the example usages at the bottom of controller.py for other ways to use it.

==================================

ABSTRACT

The ability to command a team of quadrotor drones to maintain a formation is an important capability for applications such as area surveillance, search and rescue, agriculture, and reconnaissance.  Of particular interest is operation in environments where radio and/or GPS may be denied or not sufficiently accurate for the desired application.  This is common indoors, in military scenarios due to electronic interference (jamming), and in unmapped or dynamic areas where reliance on existing maps for absolute geolocation is insufficient for relative localization of the quadcopters.  Scenarios where one or more of the quadrotors is non-cooperative, such as a seek-and-destroy type of mission, are also possible when using only visual sensors.

To address these issues, we develop and implement an algorithm that discriminates between quadrotor pixels and non-quadrotor pixels.  We implement the algorithm on a single-board ODROiD XU4 mounted to a quadrotor drone. An standard visual-band webcam is the only sensor used.

The core of our detection and localization algorithm is an hourglass convolutional neural network (CNN) that generates "heatmaps" - pixel-by-pixel likelihood estimates of the presence of a quadrotor.  A version of the Siamese networks technique is added for clutter mitigation and enforcement of temporal smoothness. Specifically we enforce the similarity of temporally adjacent frames with a Siamese loss term. The resulting centroids of each quadcopter segmentation are Kalman-filtered in time as well.

Total latency - specifically the time from frame capture to reporting the localization estimate of the observed quadcopter(s) - is kept at approximately 20ms when implmented on the ODROiD XU4. 
