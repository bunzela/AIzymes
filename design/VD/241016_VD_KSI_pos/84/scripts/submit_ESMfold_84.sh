#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_84
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/84/scripts/ESMfold_84.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/84/scripts/ESMfold_84.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/84
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/84/scripts/ESMfold_84.sh
