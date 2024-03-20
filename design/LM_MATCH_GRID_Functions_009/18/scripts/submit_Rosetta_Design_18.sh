#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_18
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/18/scripts/AI_Rosetta_Design_18.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/18/scripts/AI_Rosetta_Design_18.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/18
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/18/scripts/Rosetta_Design_18.sh
