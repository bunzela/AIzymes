#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_70
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/70/scripts/ESMfold_70.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/70/scripts/ESMfold_70.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/70
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/70/scripts/ESMfold_70.sh
