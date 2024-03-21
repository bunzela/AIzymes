#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_67
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/67/scripts/AI_Rosetta_Design_67.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/67/scripts/AI_Rosetta_Design_67.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/67
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/67/scripts/Rosetta_Design_67.sh
