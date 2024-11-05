#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_53
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/53/scripts/RosettaDesign_53.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/53/scripts/RosettaDesign_53.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/53
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/53/scripts/RosettaDesign_53.sh
