#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_17
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/17/scripts/ESMfold_17.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/17/scripts/ESMfold_17.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/17
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/17/scripts/ESMfold_17.sh
