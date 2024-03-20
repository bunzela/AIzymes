#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_0
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/0/scripts/AI_Rosetta_Design_0.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/0/scripts/AI_Rosetta_Design_0.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/0
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/0/scripts/Rosetta_Design_0.sh
