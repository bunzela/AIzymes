#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_5
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/5/scripts/AI_Rosetta_Design_5.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/5/scripts/AI_Rosetta_Design_5.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/5
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/5/scripts/Rosetta_Design_5.sh
