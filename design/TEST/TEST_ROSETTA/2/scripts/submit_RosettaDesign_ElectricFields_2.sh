#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_RosettaDesign_ElectricFields_2
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/2/scripts/RosettaDesign_ElectricFields_2.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/2/scripts/RosettaDesign_ElectricFields_2.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/2
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/2/scripts/RosettaDesign_ElectricFields_2.sh
