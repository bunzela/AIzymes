#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_95
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/95/scripts/ESMfold_95.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/95/scripts/ESMfold_95.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/95
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/95/scripts/ESMfold_95.sh
