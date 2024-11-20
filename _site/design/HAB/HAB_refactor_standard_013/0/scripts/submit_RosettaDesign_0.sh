#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_RosettaDesign_0
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/0/scripts/RosettaDesign_0.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/0/scripts/RosettaDesign_0.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/0
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/0/scripts/RosettaDesign_0.sh
