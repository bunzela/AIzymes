#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_6
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/6/scripts/RosettaDesign_6.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/6/scripts/RosettaDesign_6.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/6
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/6/scripts/RosettaDesign_6.sh
