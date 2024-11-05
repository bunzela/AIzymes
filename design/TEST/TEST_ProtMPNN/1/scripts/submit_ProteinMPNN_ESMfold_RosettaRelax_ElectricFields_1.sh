#!/bin/bash
#$ -V
#$ -cwd
#$ -N MPNN_ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_1
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/1/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_1.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/1/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_1.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/1
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/1/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_1.sh
