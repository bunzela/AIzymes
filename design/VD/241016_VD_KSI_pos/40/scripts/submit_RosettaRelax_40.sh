#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_40
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/40/scripts/RosettaRelax_40.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/40/scripts/RosettaRelax_40.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/40
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/40/scripts/RosettaRelax_40.sh
