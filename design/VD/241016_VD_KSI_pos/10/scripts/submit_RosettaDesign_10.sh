#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_10
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/10/scripts/RosettaDesign_10.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/10/scripts/RosettaDesign_10.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/10
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/10/scripts/RosettaDesign_10.sh
