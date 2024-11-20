#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_26
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/26/scripts/ESMfold_26.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/26/scripts/ESMfold_26.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/26
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/26/scripts/ESMfold_26.sh
