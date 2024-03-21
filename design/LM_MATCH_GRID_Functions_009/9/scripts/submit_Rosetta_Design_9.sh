#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_9
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/9/scripts/AI_Rosetta_Design_9.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/9/scripts/AI_Rosetta_Design_9.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/9
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/9/scripts/Rosetta_Design_9.sh
