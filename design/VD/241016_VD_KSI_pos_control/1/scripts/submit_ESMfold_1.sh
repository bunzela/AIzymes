#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_1
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/1/scripts/ESMfold_1.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/1/scripts/ESMfold_1.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/1
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/1/scripts/ESMfold_1.sh
