#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_92
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/92/scripts/RosettaDesign_92.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/92/scripts/RosettaDesign_92.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/92
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/92/scripts/RosettaDesign_92.sh
