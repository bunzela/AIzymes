#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_20
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/20/scripts/ESMfold_20.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/20/scripts/ESMfold_20.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/20
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/20/scripts/ESMfold_20.sh
