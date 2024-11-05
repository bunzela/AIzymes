#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_66
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/66/scripts/RosettaDesign_66.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/66/scripts/RosettaDesign_66.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/66
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/66/scripts/RosettaDesign_66.sh
