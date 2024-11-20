#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_115
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/115/scripts/RosettaDesign_115.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/115/scripts/RosettaDesign_115.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/115
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/115/scripts/RosettaDesign_115.sh
