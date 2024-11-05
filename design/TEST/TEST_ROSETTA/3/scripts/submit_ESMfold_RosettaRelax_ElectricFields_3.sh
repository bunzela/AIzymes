#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_ESMfold_RosettaRelax_ElectricFields_3
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/3/scripts/ESMfold_RosettaRelax_ElectricFields_3.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/3/scripts/ESMfold_RosettaRelax_ElectricFields_3.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/3
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/3/scripts/ESMfold_RosettaRelax_ElectricFields_3.sh
