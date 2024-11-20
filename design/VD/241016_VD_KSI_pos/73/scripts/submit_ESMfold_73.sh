#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_73
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/73/scripts/ESMfold_73.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/73/scripts/ESMfold_73.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/73
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/73/scripts/ESMfold_73.sh
