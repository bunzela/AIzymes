#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_36
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/36/scripts/RosettaDesign_36.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/36/scripts/RosettaDesign_36.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/36
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/36/scripts/RosettaDesign_36.sh
