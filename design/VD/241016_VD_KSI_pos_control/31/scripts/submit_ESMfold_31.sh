#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_31
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/31/scripts/ESMfold_31.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/31/scripts/ESMfold_31.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/31
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/31/scripts/ESMfold_31.sh
