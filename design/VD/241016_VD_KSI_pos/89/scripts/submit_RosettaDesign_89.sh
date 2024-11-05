#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_89
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/89/scripts/RosettaDesign_89.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/89/scripts/RosettaDesign_89.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/89
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/89/scripts/RosettaDesign_89.sh
