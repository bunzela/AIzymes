#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_84
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/84/scripts/RosettaDesign_84.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/84/scripts/RosettaDesign_84.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/84
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/84/scripts/RosettaDesign_84.sh
