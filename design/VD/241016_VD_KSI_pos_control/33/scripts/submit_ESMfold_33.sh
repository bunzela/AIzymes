#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_33
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/33/scripts/ESMfold_33.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/33/scripts/ESMfold_33.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/33
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/33/scripts/ESMfold_33.sh
