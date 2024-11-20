#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_42
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/42/scripts/RosettaDesign_42.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/42/scripts/RosettaDesign_42.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/42
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/42/scripts/RosettaDesign_42.sh
