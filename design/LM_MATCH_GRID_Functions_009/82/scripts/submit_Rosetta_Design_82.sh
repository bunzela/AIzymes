#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_82
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/82/scripts/AI_Rosetta_Design_82.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/82/scripts/AI_Rosetta_Design_82.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/82
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/82/scripts/Rosetta_Design_82.sh
