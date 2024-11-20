#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_10
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/10/scripts/ESMfold_10.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/10/scripts/ESMfold_10.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/10
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/10/scripts/ESMfold_10.sh
