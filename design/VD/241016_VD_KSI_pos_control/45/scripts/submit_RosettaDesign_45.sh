#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_45
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/45/scripts/RosettaDesign_45.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/45/scripts/RosettaDesign_45.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/45
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/45/scripts/RosettaDesign_45.sh
