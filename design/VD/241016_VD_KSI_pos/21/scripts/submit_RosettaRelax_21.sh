#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_21
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/21/scripts/RosettaRelax_21.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/21/scripts/RosettaRelax_21.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/21
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/21/scripts/RosettaRelax_21.sh
