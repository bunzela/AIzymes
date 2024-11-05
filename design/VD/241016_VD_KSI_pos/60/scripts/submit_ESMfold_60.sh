#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_60
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/60/scripts/ESMfold_60.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/60/scripts/ESMfold_60.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/60
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/60/scripts/ESMfold_60.sh
