#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_36
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/scripts/ESMfold_36.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/scripts/ESMfold_36.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/scripts/ESMfold_36.sh
