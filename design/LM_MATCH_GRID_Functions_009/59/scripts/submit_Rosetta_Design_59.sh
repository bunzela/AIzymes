#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_59
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/59/scripts/AI_Rosetta_Design_59.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/59/scripts/AI_Rosetta_Design_59.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/59
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/59/scripts/Rosetta_Design_59.sh
