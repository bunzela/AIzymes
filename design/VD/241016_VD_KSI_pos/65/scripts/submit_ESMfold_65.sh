#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_ESMfold_65
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/65/scripts/ESMfold_65.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/65/scripts/ESMfold_65.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/65
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/65/scripts/ESMfold_65.sh
