#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_34
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/34/scripts/AI_Rosetta_Design_34.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/34/scripts/AI_Rosetta_Design_34.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/34
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/34/scripts/Rosetta_Design_34.sh
