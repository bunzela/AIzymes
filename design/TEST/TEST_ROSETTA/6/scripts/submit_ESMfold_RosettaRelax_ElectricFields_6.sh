#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_ESMfold_RosettaRelax_ElectricFields_6
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/6/scripts/ESMfold_RosettaRelax_ElectricFields_6.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/6/scripts/ESMfold_RosettaRelax_ElectricFields_6.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/6
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/6/scripts/ESMfold_RosettaRelax_ElectricFields_6.sh
