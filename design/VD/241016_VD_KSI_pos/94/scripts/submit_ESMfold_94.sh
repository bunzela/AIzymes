#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_94
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/94/scripts/ESMfold_94.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/94/scripts/ESMfold_94.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/94
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/94/scripts/ESMfold_94.sh
