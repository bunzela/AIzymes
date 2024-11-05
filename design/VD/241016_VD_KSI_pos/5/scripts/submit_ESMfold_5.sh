#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_5
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/5/scripts/ESMfold_5.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/5/scripts/ESMfold_5.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/5
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/5/scripts/ESMfold_5.sh
