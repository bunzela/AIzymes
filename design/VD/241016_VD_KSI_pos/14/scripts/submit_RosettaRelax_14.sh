#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_14
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/14/scripts/RosettaRelax_14.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/14/scripts/RosettaRelax_14.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/14
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/14/scripts/RosettaRelax_14.sh
