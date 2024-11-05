#!/bin/bash
#$ -V
#$ -cwd
#$ -N mpnn_ProteinMPNN_4
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/4/scripts/ProteinMPNN_4.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/4/scripts/ProteinMPNN_4.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/4
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/4/scripts/ProteinMPNN_4.sh
