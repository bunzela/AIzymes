#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_75
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/75/scripts/AI_Rosetta_Design_75.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/75/scripts/AI_Rosetta_Design_75.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/75
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/75/scripts/Rosetta_Design_75.sh
