#!/bin/bash
#$ -V
#$ -cwd
#$ -N Ref_RosettaDesign_ElectricFields_2
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/2/scripts/RosettaDesign_ElectricFields_2.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/2/scripts/RosettaDesign_ElectricFields_2.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/2
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_standard_013/2/scripts/RosettaDesign_ElectricFields_2.sh
