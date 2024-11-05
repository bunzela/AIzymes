#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_28
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/28/scripts/RosettaDesign_28.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/28/scripts/RosettaDesign_28.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/28
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/28/scripts/RosettaDesign_28.sh
