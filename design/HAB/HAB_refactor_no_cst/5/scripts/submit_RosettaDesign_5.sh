#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_RosettaDesign_5
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_no_cst/5/scripts/RosettaDesign_5.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_no_cst/5/scripts/RosettaDesign_5.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_no_cst/5
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_no_cst/5/scripts/RosettaDesign_5.sh
