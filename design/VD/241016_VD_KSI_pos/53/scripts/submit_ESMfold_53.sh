#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_53
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/53/scripts/ESMfold_53.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/53/scripts/ESMfold_53.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/53
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/53/scripts/ESMfold_53.sh
