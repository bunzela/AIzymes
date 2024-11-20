#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_87
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/87/scripts/RosettaDesign_87.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/87/scripts/RosettaDesign_87.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/87
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/87/scripts/RosettaDesign_87.sh
