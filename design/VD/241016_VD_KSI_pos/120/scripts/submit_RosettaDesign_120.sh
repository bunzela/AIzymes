#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_120
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/120/scripts/RosettaDesign_120.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/120/scripts/RosettaDesign_120.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/120
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/120/scripts/RosettaDesign_120.sh
