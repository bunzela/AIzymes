#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_5
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/5/scripts/RosettaDesign_5.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/5/scripts/RosettaDesign_5.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/5
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/5/scripts/RosettaDesign_5.sh
