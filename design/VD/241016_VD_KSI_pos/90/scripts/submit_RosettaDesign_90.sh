#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_90
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/90/scripts/RosettaDesign_90.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/90/scripts/RosettaDesign_90.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/90
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/90/scripts/RosettaDesign_90.sh
