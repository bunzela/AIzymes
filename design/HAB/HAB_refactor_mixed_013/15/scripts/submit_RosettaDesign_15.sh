#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_15
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/15/scripts/RosettaDesign_15.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/15/scripts/RosettaDesign_15.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/15
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/15/scripts/RosettaDesign_15.sh
