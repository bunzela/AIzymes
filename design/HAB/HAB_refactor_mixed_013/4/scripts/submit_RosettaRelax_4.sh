#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaRelax_4
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/4/scripts/RosettaRelax_4.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/4/scripts/RosettaRelax_4.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/4
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/4/scripts/RosettaRelax_4.sh
