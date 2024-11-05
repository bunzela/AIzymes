#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_RosettaDesign_ElectricFields_7
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/scripts/RosettaDesign_ElectricFields_7.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/scripts/RosettaDesign_ElectricFields_7.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/scripts/RosettaDesign_ElectricFields_7.sh
