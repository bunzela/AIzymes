#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_43
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/43/scripts/ESMfold_43.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/43/scripts/ESMfold_43.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/43
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/43/scripts/ESMfold_43.sh
