#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_6
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/6/scripts/ESMfold_6.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/6/scripts/ESMfold_6.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/6
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/6/scripts/ESMfold_6.sh
