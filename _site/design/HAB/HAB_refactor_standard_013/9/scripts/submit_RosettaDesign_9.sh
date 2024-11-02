#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_RosettaDesign_9
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/9/scripts/RosettaDesign_9.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/9/scripts/RosettaDesign_9.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/9
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/9/scripts/RosettaDesign_9.sh
