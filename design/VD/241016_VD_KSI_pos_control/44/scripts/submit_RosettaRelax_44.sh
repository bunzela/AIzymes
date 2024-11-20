#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_44
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/44/scripts/RosettaRelax_44.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/44/scripts/RosettaRelax_44.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/44
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos_control/44/scripts/RosettaRelax_44.sh
