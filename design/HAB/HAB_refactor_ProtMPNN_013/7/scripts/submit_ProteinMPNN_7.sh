#!/bin/bash
#$ -V
#$ -cwd
#$ -N mpnn_ProteinMPNN_7
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/7/scripts/ProteinMPNN_7.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/7/scripts/ProteinMPNN_7.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/7
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/7/scripts/ProteinMPNN_7.sh
