#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_25
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/25/scripts/RosettaDesign_25.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/25/scripts/RosettaDesign_25.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/25
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/25/scripts/RosettaDesign_25.sh
