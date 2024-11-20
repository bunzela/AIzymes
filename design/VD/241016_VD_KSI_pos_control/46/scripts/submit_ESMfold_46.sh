#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_46
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/46/scripts/ESMfold_46.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/46/scripts/ESMfold_46.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/46
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/46/scripts/ESMfold_46.sh
