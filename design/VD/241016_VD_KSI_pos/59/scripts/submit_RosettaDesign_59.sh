#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_59
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/59/scripts/RosettaDesign_59.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/59/scripts/RosettaDesign_59.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/59
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/59/scripts/RosettaDesign_59.sh
