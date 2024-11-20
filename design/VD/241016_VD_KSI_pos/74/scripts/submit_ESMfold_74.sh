#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_74
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/74/scripts/ESMfold_74.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/74/scripts/ESMfold_74.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/74
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/74/scripts/ESMfold_74.sh
