#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_57
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/57/scripts/AI_Rosetta_Design_57.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/57/scripts/AI_Rosetta_Design_57.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/57
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/57/scripts/Rosetta_Design_57.sh
