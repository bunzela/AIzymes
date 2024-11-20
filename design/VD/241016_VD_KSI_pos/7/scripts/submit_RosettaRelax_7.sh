#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_7
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/7/scripts/RosettaRelax_7.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/7/scripts/RosettaRelax_7.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/7
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/7/scripts/RosettaRelax_7.sh
