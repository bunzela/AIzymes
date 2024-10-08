#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_ESMfold_8
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/8/scripts/ESMfold_8.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/8/scripts/ESMfold_8.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/8
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/8/scripts/ESMfold_8.sh
