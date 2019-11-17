#! /bin/bash -l
# The -l specifies that we are loading modules
#$ -pe omp 8 # using 16 processors. Available options are : 8 12 16 64
#
## Walltime limit
#$ -l h_rt=120:00:00
#
## Give the job a name.
#$ -N hC_PA_0
#
## Redirect error output to standard output
#$ -j y
#
## What project to use. 
#$ -P semslam
#
## Select a skylake processor architecture (tensorflow training runs maybe 4x faster on these for some reason)
#$ -l cpu_arch=skylake


source ~/thesis/quadcopterVision/loadModules.sh

# program name or command and its options and arguments
cd /usr3/graduate/jkdunn/thesis/quadcopterVision/
# argument list: siamese weight, 1st moment weight, 2nd moment weight, graph save location
#python run_siamese_hourglass_cnn.py 0.50 0.00 0.00 noiseFix60k_sW00p50
#python nfold_siamese_hourglass_cnn.py 0.50 0.00 0.00 biasAdd4Folds60k_sW00p50 0

python analyzePerformance.py
