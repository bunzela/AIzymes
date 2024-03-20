#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_38
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/38/scripts/AI_Rosetta_Design_38.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/38/scripts/AI_Rosetta_Design_38.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/38
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/38/scripts/Rosetta_Design_38.sh
