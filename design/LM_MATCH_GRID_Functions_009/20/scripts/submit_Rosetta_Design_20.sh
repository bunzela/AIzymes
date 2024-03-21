#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_20
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/20/scripts/AI_Rosetta_Design_20.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/20/scripts/AI_Rosetta_Design_20.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/20
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/20/scripts/Rosetta_Design_20.sh
