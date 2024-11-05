#!/bin/bash
#$ -V
#$ -cwd
#$ -N mpnn_ProteinMPNN_3
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/3/scripts/ProteinMPNN_3.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/3/scripts/ProteinMPNN_3.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/3
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/3/scripts/ProteinMPNN_3.sh
