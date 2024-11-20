#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_47
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/47/scripts/ESMfold_47.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/47/scripts/ESMfold_47.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/47
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/47/scripts/ESMfold_47.sh
