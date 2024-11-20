#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_108
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/108/scripts/RosettaDesign_108.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/108/scripts/RosettaDesign_108.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/108
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/108/scripts/RosettaDesign_108.sh
