#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_22
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/22/scripts/RosettaDesign_22.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/22/scripts/RosettaDesign_22.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/22
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/22/scripts/RosettaDesign_22.sh
