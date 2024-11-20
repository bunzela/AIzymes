#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_88
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/88/scripts/ESMfold_88.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/88/scripts/ESMfold_88.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/88
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/88/scripts/ESMfold_88.sh
