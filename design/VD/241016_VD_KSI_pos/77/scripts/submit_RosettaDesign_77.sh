#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_77
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/77/scripts/RosettaDesign_77.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/77/scripts/RosettaDesign_77.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/77
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/77/scripts/RosettaDesign_77.sh
