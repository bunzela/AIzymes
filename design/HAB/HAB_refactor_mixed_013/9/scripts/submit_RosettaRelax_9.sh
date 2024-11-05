#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaRelax_9
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/9/scripts/RosettaRelax_9.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/9/scripts/RosettaRelax_9.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/9
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/9/scripts/RosettaRelax_9.sh
