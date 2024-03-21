#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_64
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/64/scripts/AI_Rosetta_Design_64.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/64/scripts/AI_Rosetta_Design_64.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/64
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/64/scripts/Rosetta_Design_64.sh
