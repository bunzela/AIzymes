#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_91
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/91/scripts/ESMfold_91.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/91/scripts/ESMfold_91.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/91
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/91/scripts/ESMfold_91.sh
