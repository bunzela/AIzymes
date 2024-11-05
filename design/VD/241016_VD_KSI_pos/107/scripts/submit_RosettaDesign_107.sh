#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_107
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/107/scripts/RosettaDesign_107.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/107/scripts/RosettaDesign_107.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/107
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/107/scripts/RosettaDesign_107.sh
