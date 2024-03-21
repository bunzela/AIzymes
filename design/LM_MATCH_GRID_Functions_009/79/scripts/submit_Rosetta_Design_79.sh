#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_79
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/79/scripts/AI_Rosetta_Design_79.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/79/scripts/AI_Rosetta_Design_79.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/79
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/79/scripts/Rosetta_Design_79.sh
