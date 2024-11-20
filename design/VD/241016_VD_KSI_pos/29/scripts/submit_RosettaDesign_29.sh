#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_29
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/29/scripts/RosettaDesign_29.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/29/scripts/RosettaDesign_29.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/29
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/29/scripts/RosettaDesign_29.sh
