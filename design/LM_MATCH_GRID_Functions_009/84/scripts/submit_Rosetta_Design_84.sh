#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_84
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/84/scripts/AI_Rosetta_Design_84.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/84/scripts/AI_Rosetta_Design_84.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/84
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/84/scripts/Rosetta_Design_84.sh
