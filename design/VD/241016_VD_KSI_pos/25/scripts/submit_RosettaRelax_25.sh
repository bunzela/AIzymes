#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_25
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/25/scripts/RosettaRelax_25.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/25/scripts/RosettaRelax_25.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/25
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/25/scripts/RosettaRelax_25.sh
