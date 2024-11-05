#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_68
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/68/scripts/ESMfold_68.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/68/scripts/ESMfold_68.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/68
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/68/scripts/ESMfold_68.sh
