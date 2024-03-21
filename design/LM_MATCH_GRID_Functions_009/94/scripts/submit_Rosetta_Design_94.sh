#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_94
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/94/scripts/AI_Rosetta_Design_94.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/94/scripts/AI_Rosetta_Design_94.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/94
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/94/scripts/Rosetta_Design_94.sh
