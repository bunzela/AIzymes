#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_98
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/98/scripts/ESMfold_98.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/98/scripts/ESMfold_98.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/98
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/98/scripts/ESMfold_98.sh
