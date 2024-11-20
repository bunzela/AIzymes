#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_79
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/79/scripts/ESMfold_79.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/79/scripts/ESMfold_79.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/79
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/79/scripts/ESMfold_79.sh
