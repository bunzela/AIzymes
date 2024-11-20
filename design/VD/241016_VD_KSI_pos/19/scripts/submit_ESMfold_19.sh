#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_19
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/19/scripts/ESMfold_19.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/19/scripts/ESMfold_19.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/19
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/19/scripts/ESMfold_19.sh
