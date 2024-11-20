#!/bin/bash
#$ -V
#$ -cwd
#$ -N MPNN_ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_9
#$ -hard -l mf=40G
#$ -o /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/9/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_9.out
#$ -e /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/9/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_9.err

# Output folder
cd /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/9
pwd
bash /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/9/scripts/ProteinMPNN_ESMfold_RosettaRelax_ElectricFields_9.sh
