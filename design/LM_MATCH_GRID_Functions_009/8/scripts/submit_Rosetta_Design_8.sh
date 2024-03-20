#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_8
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/8/scripts/AI_Rosetta_Design_8.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/8/scripts/AI_Rosetta_Design_8.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/8
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/8/scripts/Rosetta_Design_8.sh
