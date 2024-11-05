#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_32
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32/scripts/RosettaRelax_32.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32/scripts/RosettaRelax_32.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32/scripts/RosettaRelax_32.sh
