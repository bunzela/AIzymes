#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_79
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/79/scripts/RosettaDesign_79.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/79/scripts/RosettaDesign_79.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/79
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/79/scripts/RosettaDesign_79.sh
