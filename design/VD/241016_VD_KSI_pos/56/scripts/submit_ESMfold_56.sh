#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_56
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/56/scripts/ESMfold_56.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/56/scripts/ESMfold_56.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/56
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/56/scripts/ESMfold_56.sh
