#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_RosettaDesign_8
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/8/scripts/RosettaDesign_8.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/8/scripts/RosettaDesign_8.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/8
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/8/scripts/RosettaDesign_8.sh
