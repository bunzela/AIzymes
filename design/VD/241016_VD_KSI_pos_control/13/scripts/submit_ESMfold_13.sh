#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_13
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/13/scripts/ESMfold_13.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/13/scripts/ESMfold_13.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/13
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/13/scripts/ESMfold_13.sh
