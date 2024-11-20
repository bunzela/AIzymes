#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_34
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/34/scripts/RosettaRelax_34.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/34/scripts/RosettaRelax_34.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/34
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/34/scripts/RosettaRelax_34.sh
