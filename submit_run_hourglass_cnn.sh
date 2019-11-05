#! /bin/bash -l
# The -l specifies that we are loading modules
#$ -pe omp 8 # using 16 processors. Available options are : 8 12 16 64
#
## Walltime limit
#$ -l h_rt=120:00:00
#
## Give the job a name.
#$ -N hCRS_moments
#
## Redirect error output to standard output
#$ -j y
#
## What project to use. 
#$ -P semslam

source ~/thesis/quadcopterVision/loadModules.sh

# program name or command and its options and arguments
cd /usr3/graduate/jkdunn/thesis/quadcopterVision/
python run_siamese_hourglass_cnn.py 0.0 1.0 1.0 timeRandSign_sWeight00p00_1M01p00_2M01p00
