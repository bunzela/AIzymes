#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_18
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/18/scripts/RosettaDesign_18.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/18/scripts/RosettaDesign_18.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/18
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/18/scripts/RosettaDesign_18.sh
