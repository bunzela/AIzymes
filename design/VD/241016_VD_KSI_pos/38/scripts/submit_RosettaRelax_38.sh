#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_38
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/38/scripts/RosettaRelax_38.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/38/scripts/RosettaRelax_38.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/38
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/38/scripts/RosettaRelax_38.sh
