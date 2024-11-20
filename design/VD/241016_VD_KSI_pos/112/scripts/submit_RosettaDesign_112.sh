#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_112
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/112/scripts/RosettaDesign_112.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/112/scripts/RosettaDesign_112.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/112
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/112/scripts/RosettaDesign_112.sh
