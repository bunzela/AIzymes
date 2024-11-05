#!/bin/bash
#$ -V
#$ -cwd
#$ -N ROS_ESMfold_RosettaRelax_ElectricFields_4
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/scripts/ESMfold_RosettaRelax_ElectricFields_4.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/scripts/ESMfold_RosettaRelax_ElectricFields_4.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/scripts/ESMfold_RosettaRelax_ElectricFields_4.sh
