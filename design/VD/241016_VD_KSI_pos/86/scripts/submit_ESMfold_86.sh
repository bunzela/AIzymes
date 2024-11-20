#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_86
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/86/scripts/ESMfold_86.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/86/scripts/ESMfold_86.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/86
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/86/scripts/ESMfold_86.sh
