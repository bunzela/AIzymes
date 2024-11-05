#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaRelax_7
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/scripts/RosettaRelax_7.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/scripts/RosettaRelax_7.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/scripts/RosettaRelax_7.sh
