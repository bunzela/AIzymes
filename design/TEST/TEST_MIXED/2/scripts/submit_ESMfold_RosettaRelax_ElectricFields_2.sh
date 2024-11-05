#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_ESMfold_RosettaRelax_ElectricFields_2
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/scripts/ESMfold_RosettaRelax_ElectricFields_2.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/scripts/ESMfold_RosettaRelax_ElectricFields_2.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/scripts/ESMfold_RosettaRelax_ElectricFields_2.sh
