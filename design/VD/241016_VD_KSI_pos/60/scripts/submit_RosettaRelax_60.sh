#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_60
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/60/scripts/RosettaRelax_60.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/60/scripts/RosettaRelax_60.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/60
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/60/scripts/RosettaRelax_60.sh
