#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_24
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/24/scripts/RosettaRelax_24.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/24/scripts/RosettaRelax_24.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/24
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/24/scripts/RosettaRelax_24.sh
