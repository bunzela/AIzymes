#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_55
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55/scripts/ESMfold_55.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55/scripts/ESMfold_55.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55/scripts/ESMfold_55.sh
