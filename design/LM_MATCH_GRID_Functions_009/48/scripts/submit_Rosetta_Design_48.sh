#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_48
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/48/scripts/AI_Rosetta_Design_48.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/48/scripts/AI_Rosetta_Design_48.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/48
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/48/scripts/Rosetta_Design_48.sh
