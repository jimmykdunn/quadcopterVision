#! /bin/bash -l
# The -l specifies that we are loading modules
#$ -pe omp 8 # using 16 processors. Available options are : 8 12 16 64
#
## Walltime limit
#$ -l h_rt=120:00:00
#
## Give the job a name.
#$ -N hCNN00p00
#
## Redirect error output to standard output
#$ -j y
#
## What project to use. 
#$ -P semslam

source ~/thesis/quadcopterVision/loadModules.sh

# program name or command and its options and arguments
cd /usr3/graduate/jkdunn/thesis/quadcopterVision/
python run_siamese_hourglass_cnn.py 0.00 sWeight00p00
