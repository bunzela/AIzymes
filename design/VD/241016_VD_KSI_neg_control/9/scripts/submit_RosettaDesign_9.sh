#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_9
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/9/scripts/RosettaDesign_9.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/9/scripts/RosettaDesign_9.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/9
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/9/scripts/RosettaDesign_9.sh
