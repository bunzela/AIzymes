#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_ElectricFields_1
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/1/scripts/RosettaDesign_ElectricFields_1.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/1/scripts/RosettaDesign_ElectricFields_1.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/1
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/1/scripts/RosettaDesign_ElectricFields_1.sh
