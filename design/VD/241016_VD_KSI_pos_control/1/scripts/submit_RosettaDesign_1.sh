#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_1
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/1/scripts/RosettaDesign_1.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/1/scripts/RosettaDesign_1.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/1
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/1/scripts/RosettaDesign_1.sh
