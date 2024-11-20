#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_58
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/58/scripts/RosettaRelax_58.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/58/scripts/RosettaRelax_58.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/58
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/58/scripts/RosettaRelax_58.sh
