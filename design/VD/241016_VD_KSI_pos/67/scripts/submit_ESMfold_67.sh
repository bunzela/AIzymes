#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_67
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/67/scripts/ESMfold_67.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/67/scripts/ESMfold_67.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/67
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/67/scripts/ESMfold_67.sh
