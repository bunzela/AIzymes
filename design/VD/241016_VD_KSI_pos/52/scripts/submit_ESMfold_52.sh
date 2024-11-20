#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_52
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/52/scripts/ESMfold_52.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/52/scripts/ESMfold_52.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/52
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/52/scripts/ESMfold_52.sh
