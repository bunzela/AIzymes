#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_42
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/42/scripts/ESMfold_42.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/42/scripts/ESMfold_42.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/42
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/42/scripts/ESMfold_42.sh
