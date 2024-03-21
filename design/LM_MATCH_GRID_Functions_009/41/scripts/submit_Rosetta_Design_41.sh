#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_41
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/41/scripts/AI_Rosetta_Design_41.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/41/scripts/AI_Rosetta_Design_41.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/41
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/41/scripts/Rosetta_Design_41.sh
