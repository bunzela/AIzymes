#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_85
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/85/scripts/RosettaDesign_85.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/85/scripts/RosettaDesign_85.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/85
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/85/scripts/RosettaDesign_85.sh
