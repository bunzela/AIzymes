#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_17
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/17/scripts/RosettaDesign_17.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/17/scripts/RosettaDesign_17.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/17
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/17/scripts/RosettaDesign_17.sh
