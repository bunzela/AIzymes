#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_1
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/1/scripts/RosettaDesign_1.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/1/scripts/RosettaDesign_1.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/1
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/1/scripts/RosettaDesign_1.sh
