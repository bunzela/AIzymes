#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaRelax_2
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2/scripts/RosettaRelax_2.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2/scripts/RosettaRelax_2.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2/scripts/RosettaRelax_2.sh
