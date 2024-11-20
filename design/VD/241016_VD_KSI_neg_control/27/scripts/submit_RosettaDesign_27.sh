#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_27
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/27/scripts/RosettaDesign_27.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/27/scripts/RosettaDesign_27.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/27
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/27/scripts/RosettaDesign_27.sh
