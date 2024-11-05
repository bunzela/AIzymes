#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_0
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/0/scripts/RosettaDesign_0.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/0/scripts/RosettaDesign_0.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/0
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/0/scripts/RosettaDesign_0.sh
