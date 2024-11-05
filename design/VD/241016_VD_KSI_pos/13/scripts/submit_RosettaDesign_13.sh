#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_13
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/13/scripts/RosettaDesign_13.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/13/scripts/RosettaDesign_13.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/13
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/13/scripts/RosettaDesign_13.sh
