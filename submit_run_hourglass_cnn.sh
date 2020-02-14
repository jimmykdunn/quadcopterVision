#! /bin/bash -l
# The -l specifies that we are loading modules
#$ -pe omp 2 # using 16 processors. Available options are : 8 12 16 64
#
#$ -l mem_total=125G
#
## Walltime limit
#$ -l h_rt=168:00:00
#
## Give the job a name.
#$ -N hNoHND_2_2
#
## Redirect error output to standard output
#$ -j y
#
## Notify by email when end (-m e) or abort (-m a) (and at beginning (-m b) for test)
#$ -M jkdunn@bu.edu
#$ -m e
#$ -m a
#$ -m b
#
## What project to use. 
#$ -P semslam
#
## Select a skylake processor architecture (tensorflow training runs maybe 4x faster on these for some reason)
#$ -l cpu_arch=broadwell


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
python nfold_siamese_hourglass_cnn.py 0.2 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p20 2
#python nfold_siamese_hourglass_cnn.py 0.2 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p20 3
#python nfold_siamese_hourglass_cnn.py 0.05 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p05 0
#python nfold_siamese_hourglass_cnn.py 0.05 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p05 1
#python nfold_siamese_hourglass_cnn.py 0.05 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p05 2
#python nfold_siamese_hourglass_cnn.py 0.05 0.00 0.00 biasAdd4Folds60k96x72noHand_sW00p05 3
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHandSOff3_sW00p10 0 3
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHandSOff3_sW00p10 1 3
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHandSOff3_sW00p10 2 3
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHandSOff3_sW00p10 3 3
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHandSOff5_sW00p10 0 5
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHandSOff5_sW00p10 1 5
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHandSOff5_sW00p10 2 5
#python nfold_siamese_hourglass_cnn.py 0.1 0.00 0.00 biasAdd4Folds60k96x72noHandSOff5_sW00p10 3 5

#python analyzePerformance.py
