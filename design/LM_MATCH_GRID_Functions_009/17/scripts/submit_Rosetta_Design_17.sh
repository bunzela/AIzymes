#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_17
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/17/scripts/AI_Rosetta_Design_17.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/17/scripts/AI_Rosetta_Design_17.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/17
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/17/scripts/Rosetta_Design_17.sh
