#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_12
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/12/scripts/RosettaDesign_12.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/12/scripts/RosettaDesign_12.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/12
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/12/scripts/RosettaDesign_12.sh
