#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_ESMfold_11
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/11/scripts/ESMfold_11.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/11/scripts/ESMfold_11.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/11
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/11/scripts/ESMfold_11.sh
