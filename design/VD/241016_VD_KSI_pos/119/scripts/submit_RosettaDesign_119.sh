#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_119
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/119/scripts/RosettaDesign_119.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/119/scripts/RosettaDesign_119.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/119
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/119/scripts/RosettaDesign_119.sh
