#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_16
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/16/scripts/RosettaDesign_16.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/16/scripts/RosettaDesign_16.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/16
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/16/scripts/RosettaDesign_16.sh
