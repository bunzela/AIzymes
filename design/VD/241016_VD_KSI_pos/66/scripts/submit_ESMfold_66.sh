#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_66
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/66/scripts/ESMfold_66.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/66/scripts/ESMfold_66.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/66
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/66/scripts/ESMfold_66.sh
