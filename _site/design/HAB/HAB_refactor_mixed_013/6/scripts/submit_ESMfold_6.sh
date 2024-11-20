#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_ESMfold_6
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/6/scripts/ESMfold_6.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/6/scripts/ESMfold_6.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/6
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/6/scripts/ESMfold_6.sh
