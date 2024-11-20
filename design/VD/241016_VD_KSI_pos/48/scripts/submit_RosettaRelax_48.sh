#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_48
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/48/scripts/RosettaRelax_48.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/48/scripts/RosettaRelax_48.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/48
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/48/scripts/RosettaRelax_48.sh
