#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_49
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/49/scripts/AI_Rosetta_Design_49.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/49/scripts/AI_Rosetta_Design_49.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/49
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/49/scripts/Rosetta_Design_49.sh
