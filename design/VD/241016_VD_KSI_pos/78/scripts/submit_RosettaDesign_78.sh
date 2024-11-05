#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_78
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/78/scripts/RosettaDesign_78.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/78/scripts/RosettaDesign_78.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/78
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/78/scripts/RosettaDesign_78.sh
