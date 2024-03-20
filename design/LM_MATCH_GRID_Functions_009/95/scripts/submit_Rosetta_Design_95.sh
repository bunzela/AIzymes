#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_95
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/95/scripts/AI_Rosetta_Design_95.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/95/scripts/AI_Rosetta_Design_95.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/95
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/95/scripts/Rosetta_Design_95.sh
