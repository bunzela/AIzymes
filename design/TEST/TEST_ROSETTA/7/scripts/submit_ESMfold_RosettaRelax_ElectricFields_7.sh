#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_ESMfold_RosettaRelax_ElectricFields_7
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/scripts/ESMfold_RosettaRelax_ElectricFields_7.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/scripts/ESMfold_RosettaRelax_ElectricFields_7.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/scripts/ESMfold_RosettaRelax_ElectricFields_7.sh
