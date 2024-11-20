#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_ESMfold_4
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/4/scripts/ESMfold_4.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/4/scripts/ESMfold_4.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/4
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/4/scripts/ESMfold_4.sh
