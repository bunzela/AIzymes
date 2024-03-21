#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_73
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/73/scripts/AI_Rosetta_Design_73.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/73/scripts/AI_Rosetta_Design_73.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/73
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/73/scripts/Rosetta_Design_73.sh
