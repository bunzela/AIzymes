#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_74
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/74/scripts/AI_Rosetta_Design_74.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/74/scripts/AI_Rosetta_Design_74.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/74
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/74/scripts/Rosetta_Design_74.sh
