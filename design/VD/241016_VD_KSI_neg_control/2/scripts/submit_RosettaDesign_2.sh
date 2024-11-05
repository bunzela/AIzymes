#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_2
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/2/scripts/RosettaDesign_2.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/2/scripts/RosettaDesign_2.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/2
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/2/scripts/RosettaDesign_2.sh
