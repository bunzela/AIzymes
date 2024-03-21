#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_3
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/3/scripts/AI_Rosetta_Design_3.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/3/scripts/AI_Rosetta_Design_3.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/3
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/3/scripts/Rosetta_Design_3.sh
