#!/bin/bash
#$ -V
#$ -cwd
#$ -N mpnn_ProteinMPNN_5
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/scripts/ProteinMPNN_5.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/scripts/ProteinMPNN_5.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/scripts/ProteinMPNN_5.sh
