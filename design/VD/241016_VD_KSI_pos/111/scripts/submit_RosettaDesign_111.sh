#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_111
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/111/scripts/RosettaDesign_111.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/111/scripts/RosettaDesign_111.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/111
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/111/scripts/RosettaDesign_111.sh
