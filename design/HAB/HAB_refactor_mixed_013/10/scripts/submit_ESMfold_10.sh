#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_ESMfold_10
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/10/scripts/ESMfold_10.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/10/scripts/ESMfold_10.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/10
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/10/scripts/ESMfold_10.sh
