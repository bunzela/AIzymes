#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_RosettaDesign_6
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/6/scripts/RosettaDesign_6.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/6/scripts/RosettaDesign_6.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/6
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/6/scripts/RosettaDesign_6.sh
