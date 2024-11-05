#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_88
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/88/scripts/RosettaDesign_88.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/88/scripts/RosettaDesign_88.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/88
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/88/scripts/RosettaDesign_88.sh
