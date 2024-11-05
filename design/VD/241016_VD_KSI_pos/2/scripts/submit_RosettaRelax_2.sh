#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_2
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/2/scripts/RosettaRelax_2.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/2/scripts/RosettaRelax_2.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/2
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/2/scripts/RosettaRelax_2.sh
