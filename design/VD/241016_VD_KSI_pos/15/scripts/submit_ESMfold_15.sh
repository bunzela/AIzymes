#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_15
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/15/scripts/ESMfold_15.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/15/scripts/ESMfold_15.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/15
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/15/scripts/ESMfold_15.sh
