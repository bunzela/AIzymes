#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_ESMfold_5
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/5/scripts/ESMfold_5.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/5/scripts/ESMfold_5.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/5
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/5/scripts/ESMfold_5.sh
