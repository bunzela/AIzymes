#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_10
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/10/scripts/RosettaRelax_10.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/10/scripts/RosettaRelax_10.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/10
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/10/scripts/RosettaRelax_10.sh
