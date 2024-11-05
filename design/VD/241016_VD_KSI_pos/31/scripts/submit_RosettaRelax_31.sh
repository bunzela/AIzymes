#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_31
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/31/scripts/RosettaRelax_31.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/31/scripts/RosettaRelax_31.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/31
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/31/scripts/RosettaRelax_31.sh
