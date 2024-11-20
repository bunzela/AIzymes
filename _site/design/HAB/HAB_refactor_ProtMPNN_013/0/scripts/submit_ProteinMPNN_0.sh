#!/bin/bash
#$ -V
#$ -cwd
#$ -N mpnn_ProteinMPNN_0
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/0/scripts/ProteinMPNN_0.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/0/scripts/ProteinMPNN_0.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/0
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/0/scripts/ProteinMPNN_0.sh
