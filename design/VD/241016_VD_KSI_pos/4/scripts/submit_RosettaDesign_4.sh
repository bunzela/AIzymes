#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_4
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/4/scripts/RosettaDesign_4.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/4/scripts/RosettaDesign_4.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/4
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/4/scripts/RosettaDesign_4.sh
