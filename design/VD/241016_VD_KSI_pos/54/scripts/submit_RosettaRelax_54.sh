#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_54
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/54/scripts/RosettaRelax_54.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/54/scripts/RosettaRelax_54.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/54
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/54/scripts/RosettaRelax_54.sh
