#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_55
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55/scripts/RosettaRelax_55.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55/scripts/RosettaRelax_55.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/55/scripts/RosettaRelax_55.sh
