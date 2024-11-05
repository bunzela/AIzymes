#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_71
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/71/scripts/RosettaDesign_71.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/71/scripts/RosettaDesign_71.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/71
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/71/scripts/RosettaDesign_71.sh
