#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_46
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/46/scripts/RosettaDesign_46.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/46/scripts/RosettaDesign_46.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/46
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/46/scripts/RosettaDesign_46.sh
