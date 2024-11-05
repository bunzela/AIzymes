#!/bin/bash
#$ -V
#$ -cwd
#$ -N Ref_RosettaDesign_ElectricFields_3
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/3/scripts/RosettaDesign_ElectricFields_3.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/3/scripts/RosettaDesign_ElectricFields_3.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/3
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/3/scripts/RosettaDesign_ElectricFields_3.sh
