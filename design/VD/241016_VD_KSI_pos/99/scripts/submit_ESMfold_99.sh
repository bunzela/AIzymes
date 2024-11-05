#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_99
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/99/scripts/ESMfold_99.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/99/scripts/ESMfold_99.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/99
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/99/scripts/ESMfold_99.sh
