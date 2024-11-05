#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_50
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/50/scripts/RosettaRelax_50.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/50/scripts/RosettaRelax_50.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/50
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/50/scripts/RosettaRelax_50.sh
