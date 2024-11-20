#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_32
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/32/scripts/ESMfold_32.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/32/scripts/ESMfold_32.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/32
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/32/scripts/ESMfold_32.sh
