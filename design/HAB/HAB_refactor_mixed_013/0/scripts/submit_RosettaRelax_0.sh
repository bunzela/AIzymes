#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaRelax_0
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/0/scripts/RosettaRelax_0.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/0/scripts/RosettaRelax_0.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/0
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/0/scripts/RosettaRelax_0.sh
