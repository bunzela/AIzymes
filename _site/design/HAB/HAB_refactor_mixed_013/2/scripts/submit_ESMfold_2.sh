#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_ESMfold_2
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2/scripts/ESMfold_2.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2/scripts/ESMfold_2.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2/scripts/ESMfold_2.sh
