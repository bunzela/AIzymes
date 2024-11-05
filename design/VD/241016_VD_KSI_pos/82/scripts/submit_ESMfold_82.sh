#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_82
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/82/scripts/ESMfold_82.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/82/scripts/ESMfold_82.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/82
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/82/scripts/ESMfold_82.sh
