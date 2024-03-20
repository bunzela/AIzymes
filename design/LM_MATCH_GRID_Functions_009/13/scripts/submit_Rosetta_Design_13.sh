#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_13
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/13/scripts/AI_Rosetta_Design_13.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/13/scripts/AI_Rosetta_Design_13.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/13
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/13/scripts/Rosetta_Design_13.sh
