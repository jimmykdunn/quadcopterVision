#! /bin/bash -l
# The -l specifies that we are loading modules
#$ -pe omp 2 # using 16 processors. Available options are : 8 12 16 64
#
## Walltime limit
#$ -l h_rt=96:00:00
#
## Give the job a name.
#$ -N hNoHND_5_3
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

#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p10 0
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p10 1
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p10 2
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p10 3
#python nfold_siamese_hourglass_cnn.py 0.0 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p00 0
#python nfold_siamese_hourglass_cnn.py 0.0 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p00 1
#python nfold_siamese_hourglass_cnn.py 0.0 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p00 2
#python nfold_siamese_hourglass_cnn.py 0.0 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p00 3
#python nfold_siamese_hourglass_cnn.py 0.2 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p20 0
#python nfold_siamese_hourglass_cnn.py 0.2 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p20 1
#python nfold_siamese_hourglass_cnn.py 0.2 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p20 2
#python nfold_siamese_hourglass_cnn.py 0.2 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p20 3
#python nfold_siamese_hourglass_cnn.py 0.05 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p05 0
#python nfold_siamese_hourglass_cnn.py 0.05 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p05 1
#python nfold_siamese_hourglass_cnn.py 0.05 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p05 2
python nfold_siamese_hourglass_cnn.py 0.05 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p05 3
#python nfold_siamese_hourglass_cnn.py 0.01 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p01 0
#python nfold_siamese_hourglass_cnn.py 0.01 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p01 1
#python nfold_siamese_hourglass_cnn.py 0.01 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p01 2
#python nfold_siamese_hourglass_cnn.py 0.01 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p01 3
#python nfold_siamese_hourglass_cnn.py 0.5 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p50 0
#python nfold_siamese_hourglass_cnn.py 0.5 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p50 1
#python nfold_siamese_hourglass_cnn.py 0.5 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p50 2
#python nfold_siamese_hourglass_cnn.py 0.5 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p50 3

#python analyzePerformance.py
