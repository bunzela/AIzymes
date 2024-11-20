#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_69
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/69/scripts/RosettaDesign_69.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/69/scripts/RosettaDesign_69.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/69
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/69/scripts/RosettaDesign_69.sh
