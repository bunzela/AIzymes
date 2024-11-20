#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_5
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/5/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_5.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/5/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_5.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/5
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/5/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_5.sh
