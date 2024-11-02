#!/bin/bash
#$ -V
#$ -cwd
#$ -N mpnn_ProteinMPNN_8
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/8/scripts/ProteinMPNN_8.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/8/scripts/ProteinMPNN_8.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/8
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/8/scripts/ProteinMPNN_8.sh
