#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_7
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/7/scripts/AI_Rosetta_Design_7.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/7/scripts/AI_Rosetta_Design_7.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/7
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/7/scripts/Rosetta_Design_7.sh
