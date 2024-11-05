#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_54
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/54/scripts/ESMfold_54.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/54/scripts/ESMfold_54.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/54
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/54/scripts/ESMfold_54.sh
