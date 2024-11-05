#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_41
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/41/scripts/ESMfold_41.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/41/scripts/ESMfold_41.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/41
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/41/scripts/ESMfold_41.sh
