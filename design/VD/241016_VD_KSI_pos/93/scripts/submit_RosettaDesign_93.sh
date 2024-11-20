#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaDesign_93
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/93/scripts/RosettaDesign_93.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/93/scripts/RosettaDesign_93.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/93
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/93/scripts/RosettaDesign_93.sh
