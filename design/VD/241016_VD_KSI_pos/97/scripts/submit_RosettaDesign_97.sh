#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_97
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/97/scripts/RosettaDesign_97.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/97/scripts/RosettaDesign_97.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/97
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/97/scripts/RosettaDesign_97.sh
