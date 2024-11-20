#!/bin/bash
#$ -V
#$ -cwd
#$ -N KSI_pos_RosettaRelax_33
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/33/scripts/RosettaRelax_33.out
#$ -e /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/33/scripts/RosettaRelax_33.err

# Output folder
cd /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/33
pwd
bash /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/33/scripts/RosettaRelax_33.sh
