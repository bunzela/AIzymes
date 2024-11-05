#!/bin/bash
#$ -V
#$ -cwd
#$ -N mpnn_ProteinMPNN_9
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/9/scripts/ProteinMPNN_9.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/9/scripts/ProteinMPNN_9.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/9
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/9/scripts/ProteinMPNN_9.sh
