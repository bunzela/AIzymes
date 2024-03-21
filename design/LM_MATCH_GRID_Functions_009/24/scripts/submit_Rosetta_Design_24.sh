#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_24
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/24/scripts/AI_Rosetta_Design_24.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/24/scripts/AI_Rosetta_Design_24.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/24
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/24/scripts/Rosetta_Design_24.sh
