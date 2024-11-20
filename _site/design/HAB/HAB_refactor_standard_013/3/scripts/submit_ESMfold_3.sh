#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_ESMfold_3
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/3/scripts/ESMfold_3.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/3/scripts/ESMfold_3.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/3
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/3/scripts/ESMfold_3.sh
