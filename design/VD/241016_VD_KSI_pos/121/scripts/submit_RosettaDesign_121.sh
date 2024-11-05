#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_121
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/121/scripts/RosettaDesign_121.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/121/scripts/RosettaDesign_121.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/121
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/121/scripts/RosettaDesign_121.sh
