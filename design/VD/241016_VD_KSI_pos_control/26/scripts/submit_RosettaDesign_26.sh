#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_26
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/26/scripts/RosettaDesign_26.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/26/scripts/RosettaDesign_26.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/26
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/26/scripts/RosettaDesign_26.sh
