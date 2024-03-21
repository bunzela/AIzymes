#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_80
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/80/scripts/AI_Rosetta_Design_80.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/80/scripts/AI_Rosetta_Design_80.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/80
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/80/scripts/Rosetta_Design_80.sh
