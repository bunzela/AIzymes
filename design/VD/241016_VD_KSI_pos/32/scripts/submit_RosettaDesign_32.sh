#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_32
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32/scripts/RosettaDesign_32.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32/scripts/RosettaDesign_32.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32/scripts/RosettaDesign_32.sh
