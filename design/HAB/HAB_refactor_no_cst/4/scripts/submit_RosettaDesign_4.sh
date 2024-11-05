#!/bin/bash
#$ -V
#$ -cwd
#$ -N REF_RosettaDesign_4
#$ -hard -l mf=10G
#$ -o /home/bunzelh/AIzymes/design/HAB/HAB_refactor_no_cst/4/scripts/RosettaDesign_4.out
#$ -e /home/bunzelh/AIzymes/design/HAB/HAB_refactor_no_cst/4/scripts/RosettaDesign_4.err

# Output folder
cd /home/bunzelh/AIzymes/design/HAB/HAB_refactor_no_cst/4
pwd
bash /home/bunzelh/AIzymes/design/HAB/HAB_refactor_no_cst/4/scripts/RosettaDesign_4.sh
