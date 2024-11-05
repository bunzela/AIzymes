#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_57
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/57/scripts/RosettaRelax_57.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/57/scripts/RosettaRelax_57.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/57
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/57/scripts/RosettaRelax_57.sh
