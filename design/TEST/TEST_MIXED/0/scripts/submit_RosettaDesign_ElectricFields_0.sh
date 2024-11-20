#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_ElectricFields_0
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/0/scripts/RosettaDesign_ElectricFields_0.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/0/scripts/RosettaDesign_ElectricFields_0.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/0
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/0/scripts/RosettaDesign_ElectricFields_0.sh
