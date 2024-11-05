#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_52
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/52/scripts/RosettaDesign_52.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/52/scripts/RosettaDesign_52.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/52
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/52/scripts/RosettaDesign_52.sh
