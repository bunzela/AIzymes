#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_61
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/61/scripts/RosettaRelax_61.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/61/scripts/RosettaRelax_61.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/61
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/61/scripts/RosettaRelax_61.sh
