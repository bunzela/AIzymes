#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_RosettaDesign_2
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/2/scripts/RosettaDesign_2.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/2/scripts/RosettaDesign_2.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/2
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/2/scripts/RosettaDesign_2.sh
