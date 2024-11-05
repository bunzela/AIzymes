#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_35
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/35/scripts/RosettaRelax_35.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/35/scripts/RosettaRelax_35.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/35
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/35/scripts/RosettaRelax_35.sh
