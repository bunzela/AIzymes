#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_neg_RosettaDesign_20
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/20/scripts/RosettaDesign_20.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/20/scripts/RosettaDesign_20.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/20
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_neg_control/20/scripts/RosettaDesign_20.sh
