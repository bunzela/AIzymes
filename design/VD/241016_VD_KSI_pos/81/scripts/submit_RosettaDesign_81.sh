#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_81
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/81/scripts/RosettaDesign_81.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/81/scripts/RosettaDesign_81.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/81
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/81/scripts/RosettaDesign_81.sh
