#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_60
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/60/scripts/AI_Rosetta_Design_60.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/60/scripts/AI_Rosetta_Design_60.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/60
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/60/scripts/Rosetta_Design_60.sh
