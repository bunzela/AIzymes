#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_51
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/51/scripts/ESMfold_51.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/51/scripts/ESMfold_51.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/51
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/51/scripts/ESMfold_51.sh
