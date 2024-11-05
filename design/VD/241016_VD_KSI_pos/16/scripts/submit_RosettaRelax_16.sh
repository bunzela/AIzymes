#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_16
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/16/scripts/RosettaRelax_16.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/16/scripts/RosettaRelax_16.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/16
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/16/scripts/RosettaRelax_16.sh
