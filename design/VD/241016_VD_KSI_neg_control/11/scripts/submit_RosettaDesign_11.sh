#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_11
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/11/scripts/RosettaDesign_11.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/11/scripts/RosettaDesign_11.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/11
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/11/scripts/RosettaDesign_11.sh
