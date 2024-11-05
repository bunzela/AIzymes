#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_80
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/80/scripts/RosettaDesign_80.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/80/scripts/RosettaDesign_80.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/80
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/80/scripts/RosettaDesign_80.sh
