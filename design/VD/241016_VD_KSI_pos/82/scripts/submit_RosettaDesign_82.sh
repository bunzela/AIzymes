#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_82
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/82/scripts/RosettaDesign_82.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/82/scripts/RosettaDesign_82.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/82
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/82/scripts/RosettaDesign_82.sh
