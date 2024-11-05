#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_2
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/2/scripts/ESMfold_2.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/2/scripts/ESMfold_2.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/2
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/2/scripts/ESMfold_2.sh
