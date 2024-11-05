#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_28
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/28/scripts/ESMfold_28.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/28/scripts/ESMfold_28.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/28
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/28/scripts/ESMfold_28.sh
