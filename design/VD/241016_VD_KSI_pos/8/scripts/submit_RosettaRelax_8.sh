#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_8
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/8/scripts/RosettaRelax_8.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/8/scripts/RosettaRelax_8.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/8
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/8/scripts/RosettaRelax_8.sh
