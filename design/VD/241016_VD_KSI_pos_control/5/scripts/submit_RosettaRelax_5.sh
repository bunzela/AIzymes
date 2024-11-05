#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_5
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/5/scripts/RosettaRelax_5.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/5/scripts/RosettaRelax_5.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/5
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/5/scripts/RosettaRelax_5.sh
