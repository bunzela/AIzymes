#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_RosettaDesign_ElectricFields_6
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/6/scripts/RosettaDesign_ElectricFields_6.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/6/scripts/RosettaDesign_ElectricFields_6.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/6
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/6/scripts/RosettaDesign_ElectricFields_6.sh
