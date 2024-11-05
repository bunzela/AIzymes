#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_RosettaDesign_ElectricFields_0
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/0/scripts/RosettaDesign_ElectricFields_0.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/0/scripts/RosettaDesign_ElectricFields_0.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/0
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/0/scripts/RosettaDesign_ElectricFields_0.sh
