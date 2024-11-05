#!/bin/bash
#$ -V
#$ -cwd
#$ -N mpnn_ProteinMPNN_1
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/1/scripts/ProteinMPNN_1.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/1/scripts/ProteinMPNN_1.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/1
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/1/scripts/ProteinMPNN_1.sh
