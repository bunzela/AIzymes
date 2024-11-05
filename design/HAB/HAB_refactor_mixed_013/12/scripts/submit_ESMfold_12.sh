#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_ESMfold_12
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/12/scripts/ESMfold_12.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/12/scripts/ESMfold_12.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/12
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/12/scripts/ESMfold_12.sh
