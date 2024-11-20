#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_23
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/23/scripts/RosettaDesign_23.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/23/scripts/RosettaDesign_23.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/23
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/23/scripts/RosettaDesign_23.sh
