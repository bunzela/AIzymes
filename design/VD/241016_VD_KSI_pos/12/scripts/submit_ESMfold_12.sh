#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_12
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/12/scripts/ESMfold_12.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/12/scripts/ESMfold_12.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/12
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/12/scripts/ESMfold_12.sh
