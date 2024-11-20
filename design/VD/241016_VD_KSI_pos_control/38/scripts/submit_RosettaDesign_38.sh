#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_38
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/38/scripts/RosettaDesign_38.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/38/scripts/RosettaDesign_38.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/38
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/38/scripts/RosettaDesign_38.sh
