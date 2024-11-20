#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_80
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/80/scripts/ESMfold_80.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/80/scripts/ESMfold_80.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/80
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/80/scripts/ESMfold_80.sh
