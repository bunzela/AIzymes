#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_ESMfold_RosettaRelax_ElectricFields_1
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/1/scripts/ESMfold_RosettaRelax_ElectricFields_1.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/1/scripts/ESMfold_RosettaRelax_ElectricFields_1.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/1
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/1/scripts/ESMfold_RosettaRelax_ElectricFields_1.sh
