#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_90
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/90/scripts/ESMfold_90.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/90/scripts/ESMfold_90.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/90
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/90/scripts/ESMfold_90.sh
