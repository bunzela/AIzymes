#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_42
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/42/scripts/AI_Rosetta_Design_42.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/42/scripts/AI_Rosetta_Design_42.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/42
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/42/scripts/Rosetta_Design_42.sh
