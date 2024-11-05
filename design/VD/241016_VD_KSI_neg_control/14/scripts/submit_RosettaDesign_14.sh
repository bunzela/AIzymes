#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_14
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/14/scripts/RosettaDesign_14.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/14/scripts/RosettaDesign_14.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/14
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/14/scripts/RosettaDesign_14.sh
