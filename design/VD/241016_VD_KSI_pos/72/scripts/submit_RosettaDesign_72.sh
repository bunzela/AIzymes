#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_72
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/72/scripts/RosettaDesign_72.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/72/scripts/RosettaDesign_72.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/72
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/72/scripts/RosettaDesign_72.sh
