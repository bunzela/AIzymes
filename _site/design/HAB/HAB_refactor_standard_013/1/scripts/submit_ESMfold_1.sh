#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_ESMfold_1
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/1/scripts/ESMfold_1.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/1/scripts/ESMfold_1.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/1
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/1/scripts/ESMfold_1.sh
