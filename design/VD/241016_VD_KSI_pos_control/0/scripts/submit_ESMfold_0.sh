#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_0
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/0/scripts/ESMfold_0.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/0/scripts/ESMfold_0.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/0
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/0/scripts/ESMfold_0.sh
