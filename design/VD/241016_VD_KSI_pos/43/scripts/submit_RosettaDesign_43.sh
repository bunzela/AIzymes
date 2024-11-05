#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_43
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/43/scripts/RosettaDesign_43.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/43/scripts/RosettaDesign_43.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/43
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/43/scripts/RosettaDesign_43.sh
