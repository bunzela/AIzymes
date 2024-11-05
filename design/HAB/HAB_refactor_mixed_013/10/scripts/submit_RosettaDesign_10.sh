#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_10
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/10/scripts/RosettaDesign_10.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/10/scripts/RosettaDesign_10.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/10
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/10/scripts/RosettaDesign_10.sh
