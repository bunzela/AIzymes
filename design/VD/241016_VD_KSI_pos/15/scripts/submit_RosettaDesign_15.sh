#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_15
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/15/scripts/RosettaDesign_15.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/15/scripts/RosettaDesign_15.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/15
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/15/scripts/RosettaDesign_15.sh
