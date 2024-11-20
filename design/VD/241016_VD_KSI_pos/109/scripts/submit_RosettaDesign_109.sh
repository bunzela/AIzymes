#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_109
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/109/scripts/RosettaDesign_109.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/109/scripts/RosettaDesign_109.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/109
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/109/scripts/RosettaDesign_109.sh
