#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_110
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/110/scripts/RosettaDesign_110.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/110/scripts/RosettaDesign_110.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/110
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/110/scripts/RosettaDesign_110.sh
