#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_16
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/16/scripts/RosettaDesign_16.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/16/scripts/RosettaDesign_16.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/16
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/16/scripts/RosettaDesign_16.sh
