#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_RosettaDesign_ElectricFields_5
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/5/scripts/RosettaDesign_ElectricFields_5.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/5/scripts/RosettaDesign_ElectricFields_5.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/5
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/5/scripts/RosettaDesign_ElectricFields_5.sh
