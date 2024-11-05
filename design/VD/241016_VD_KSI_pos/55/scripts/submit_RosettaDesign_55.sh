#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_55
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55/scripts/RosettaDesign_55.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55/scripts/RosettaDesign_55.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55/scripts/RosettaDesign_55.sh
