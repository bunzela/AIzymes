#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_8
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/8/scripts/RosettaDesign_8.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/8/scripts/RosettaDesign_8.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/8
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/8/scripts/RosettaDesign_8.sh
