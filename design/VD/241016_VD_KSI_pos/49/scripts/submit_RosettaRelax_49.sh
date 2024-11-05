#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_49
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/49/scripts/RosettaRelax_49.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/49/scripts/RosettaRelax_49.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/49
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/49/scripts/RosettaRelax_49.sh
