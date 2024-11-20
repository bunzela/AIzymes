#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_76
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/76/scripts/ESMfold_76.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/76/scripts/ESMfold_76.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/76
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/76/scripts/ESMfold_76.sh
