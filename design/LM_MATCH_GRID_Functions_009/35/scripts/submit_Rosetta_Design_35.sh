#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_35
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/35/scripts/AI_Rosetta_Design_35.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/35/scripts/AI_Rosetta_Design_35.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/35
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/35/scripts/Rosetta_Design_35.sh
