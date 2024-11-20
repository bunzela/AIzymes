#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_39
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/39/scripts/ESMfold_39.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/39/scripts/ESMfold_39.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/39
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/39/scripts/ESMfold_39.sh
