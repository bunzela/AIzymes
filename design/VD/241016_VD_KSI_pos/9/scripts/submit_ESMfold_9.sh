#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_9
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/9/scripts/ESMfold_9.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/9/scripts/ESMfold_9.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/9
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/9/scripts/ESMfold_9.sh
