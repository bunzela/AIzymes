#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_114
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/114/scripts/RosettaDesign_114.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/114/scripts/RosettaDesign_114.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/114
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/114/scripts/RosettaDesign_114.sh
