#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_30
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/30/scripts/RosettaRelax_30.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/30/scripts/RosettaRelax_30.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/30
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/30/scripts/RosettaRelax_30.sh
