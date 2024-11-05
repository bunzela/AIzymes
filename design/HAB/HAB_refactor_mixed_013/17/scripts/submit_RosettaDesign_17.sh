#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_17
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/17/scripts/RosettaDesign_17.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/17/scripts/RosettaDesign_17.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/17
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/17/scripts/RosettaDesign_17.sh
