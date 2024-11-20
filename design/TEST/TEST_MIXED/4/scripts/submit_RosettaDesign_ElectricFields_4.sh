#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_RosettaDesign_ElectricFields_4
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/4/scripts/RosettaDesign_ElectricFields_4.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/4/scripts/RosettaDesign_ElectricFields_4.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/4
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/4/scripts/RosettaDesign_ElectricFields_4.sh
