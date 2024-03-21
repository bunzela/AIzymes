#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_22
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/22/scripts/AI_Rosetta_Design_22.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/22/scripts/AI_Rosetta_Design_22.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/22
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/22/scripts/Rosetta_Design_22.sh
