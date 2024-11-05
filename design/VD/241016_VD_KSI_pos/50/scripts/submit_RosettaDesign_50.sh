#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_50
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/50/scripts/RosettaDesign_50.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/50/scripts/RosettaDesign_50.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/50
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/50/scripts/RosettaDesign_50.sh
