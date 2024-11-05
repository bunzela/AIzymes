#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_75
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/75/scripts/ESMfold_75.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/75/scripts/ESMfold_75.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/75
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/75/scripts/ESMfold_75.sh
