#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_37
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/37/scripts/ESMfold_37.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/37/scripts/ESMfold_37.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/37
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/37/scripts/ESMfold_37.sh
