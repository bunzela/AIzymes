#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_24
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/24/scripts/RosettaDesign_24.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/24/scripts/RosettaDesign_24.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/24
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/24/scripts/RosettaDesign_24.sh
