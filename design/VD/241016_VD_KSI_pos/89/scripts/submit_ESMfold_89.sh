#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_89
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/89/scripts/ESMfold_89.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/89/scripts/ESMfold_89.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/89
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/89/scripts/ESMfold_89.sh
