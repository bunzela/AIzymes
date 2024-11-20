#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_92
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/92/scripts/ESMfold_92.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/92/scripts/ESMfold_92.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/92
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/92/scripts/ESMfold_92.sh
