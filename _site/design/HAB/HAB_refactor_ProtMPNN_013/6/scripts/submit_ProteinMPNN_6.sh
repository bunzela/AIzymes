#!/bin/bash
#$ -V
#$ -cwd
#$ -N mpnn_ProteinMPNN_6
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/6/scripts/ProteinMPNN_6.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/6/scripts/ProteinMPNN_6.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/6
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/6/scripts/ProteinMPNN_6.sh
