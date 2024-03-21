#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_1
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/1/scripts/AI_Rosetta_Design_1.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/1/scripts/AI_Rosetta_Design_1.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/1
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/1/scripts/Rosetta_Design_1.sh
