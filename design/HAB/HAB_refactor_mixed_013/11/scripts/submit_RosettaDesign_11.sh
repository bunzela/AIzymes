#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_11
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/11/scripts/RosettaDesign_11.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/11/scripts/RosettaDesign_11.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/11
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/11/scripts/RosettaDesign_11.sh
