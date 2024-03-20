#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_89
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/89/scripts/AI_Rosetta_Design_89.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/89/scripts/AI_Rosetta_Design_89.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/89
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/89/scripts/Rosetta_Design_89.sh
