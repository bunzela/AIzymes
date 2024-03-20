#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_85
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/85/scripts/AI_Rosetta_Design_85.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/85/scripts/AI_Rosetta_Design_85.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/85
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/85/scripts/Rosetta_Design_85.sh
