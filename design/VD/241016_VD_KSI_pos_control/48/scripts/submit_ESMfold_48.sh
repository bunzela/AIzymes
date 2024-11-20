#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_48
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/48/scripts/ESMfold_48.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/48/scripts/ESMfold_48.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/48
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/48/scripts/ESMfold_48.sh
