#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_17
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/17/scripts/RosettaRelax_17.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/17/scripts/RosettaRelax_17.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/17
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/17/scripts/RosettaRelax_17.sh
