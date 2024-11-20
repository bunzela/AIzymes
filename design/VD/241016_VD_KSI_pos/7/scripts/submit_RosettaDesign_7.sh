#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_7
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/7/scripts/RosettaDesign_7.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/7/scripts/RosettaDesign_7.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/7
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/7/scripts/RosettaDesign_7.sh
