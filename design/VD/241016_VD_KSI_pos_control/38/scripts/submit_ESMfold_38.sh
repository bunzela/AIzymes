#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_38
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/38/scripts/ESMfold_38.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/38/scripts/ESMfold_38.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/38
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/38/scripts/ESMfold_38.sh
