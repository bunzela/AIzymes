#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_RosettaDesign_7
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/7/scripts/RosettaDesign_7.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/7/scripts/RosettaDesign_7.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/7
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/7/scripts/RosettaDesign_7.sh
