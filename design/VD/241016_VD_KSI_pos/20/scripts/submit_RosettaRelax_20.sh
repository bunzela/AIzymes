#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_20
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/20/scripts/RosettaRelax_20.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/20/scripts/RosettaRelax_20.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/20
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/20/scripts/RosettaRelax_20.sh
