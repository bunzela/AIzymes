#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_14
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/14/scripts/ESMfold_14.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/14/scripts/ESMfold_14.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/14
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/14/scripts/ESMfold_14.sh
