#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_62
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/62/scripts/AI_Rosetta_Design_62.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/62/scripts/AI_Rosetta_Design_62.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/62
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/62/scripts/Rosetta_Design_62.sh
