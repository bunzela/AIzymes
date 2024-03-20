#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_66
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/66/scripts/AI_Rosetta_Design_66.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/66/scripts/AI_Rosetta_Design_66.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/66
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/66/scripts/Rosetta_Design_66.sh
