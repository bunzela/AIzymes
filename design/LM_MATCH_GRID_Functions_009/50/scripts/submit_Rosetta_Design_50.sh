#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_50
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/50/scripts/AI_Rosetta_Design_50.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/50/scripts/AI_Rosetta_Design_50.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/50
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/50/scripts/Rosetta_Design_50.sh
