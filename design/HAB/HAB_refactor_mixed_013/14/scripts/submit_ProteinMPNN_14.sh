#!/bin/bash
#$ -V
#$ -cwd
#$ -N MIX_ProteinMPNN_14
#$ -hard -l mf=20G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/14/scripts/ProteinMPNN_14.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/14/scripts/ProteinMPNN_14.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/14
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/14/scripts/ProteinMPNN_14.sh
