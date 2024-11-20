#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_29
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/29/scripts/ESMfold_29.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/29/scripts/ESMfold_29.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/29
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/29/scripts/ESMfold_29.sh
