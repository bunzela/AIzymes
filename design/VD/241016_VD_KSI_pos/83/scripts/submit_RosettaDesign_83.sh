#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_83
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/83/scripts/RosettaDesign_83.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/83/scripts/RosettaDesign_83.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/83
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/83/scripts/RosettaDesign_83.sh
