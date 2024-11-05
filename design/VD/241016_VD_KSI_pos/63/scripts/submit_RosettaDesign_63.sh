#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_63
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/63/scripts/RosettaDesign_63.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/63/scripts/RosettaDesign_63.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/63
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/63/scripts/RosettaDesign_63.sh
