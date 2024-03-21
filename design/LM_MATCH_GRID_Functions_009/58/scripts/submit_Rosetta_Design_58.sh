#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_58
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/58/scripts/AI_Rosetta_Design_58.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/58/scripts/AI_Rosetta_Design_58.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/58
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/58/scripts/Rosetta_Design_58.sh
