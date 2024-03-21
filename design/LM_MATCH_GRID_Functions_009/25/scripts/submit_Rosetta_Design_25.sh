#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_25
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/25/scripts/AI_Rosetta_Design_25.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/25/scripts/AI_Rosetta_Design_25.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/25
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/25/scripts/Rosetta_Design_25.sh
