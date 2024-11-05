#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_13
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/13/scripts/RosettaDesign_13.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/13/scripts/RosettaDesign_13.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/13
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/13/scripts/RosettaDesign_13.sh
