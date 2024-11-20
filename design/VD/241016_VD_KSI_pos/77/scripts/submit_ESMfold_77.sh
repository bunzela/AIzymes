#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_77
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/77/scripts/ESMfold_77.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/77/scripts/ESMfold_77.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/77
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/77/scripts/ESMfold_77.sh
