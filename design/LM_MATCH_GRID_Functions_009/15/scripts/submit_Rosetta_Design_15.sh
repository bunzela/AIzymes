#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_15
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/15/scripts/AI_Rosetta_Design_15.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/15/scripts/AI_Rosetta_Design_15.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/15
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/15/scripts/Rosetta_Design_15.sh
