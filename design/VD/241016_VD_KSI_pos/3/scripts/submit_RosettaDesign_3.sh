#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_3
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/3/scripts/RosettaDesign_3.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/3/scripts/RosettaDesign_3.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/3
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/3/scripts/RosettaDesign_3.sh
