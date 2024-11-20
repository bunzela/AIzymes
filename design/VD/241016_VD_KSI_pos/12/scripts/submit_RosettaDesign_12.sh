#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_12
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/12/scripts/RosettaDesign_12.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/12/scripts/RosettaDesign_12.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/12
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/12/scripts/RosettaDesign_12.sh
