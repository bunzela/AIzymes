#!/bin/bash
#$ -V
#$ -cwd
#$ -N MPNN_ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_0
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/0/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_0.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/0/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_0.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/0
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/0/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_0.sh
