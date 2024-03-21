#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_51
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/51/scripts/AI_Rosetta_Design_51.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/51/scripts/AI_Rosetta_Design_51.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/51
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/51/scripts/Rosetta_Design_51.sh
