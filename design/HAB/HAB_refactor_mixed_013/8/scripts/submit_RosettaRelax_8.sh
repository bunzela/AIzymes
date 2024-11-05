#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaRelax_8
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/8/scripts/RosettaRelax_8.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/8/scripts/RosettaRelax_8.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/8
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/8/scripts/RosettaRelax_8.sh
