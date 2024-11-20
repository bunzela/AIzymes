#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_45
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/45/scripts/ESMfold_45.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/45/scripts/ESMfold_45.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/45
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/45/scripts/ESMfold_45.sh
