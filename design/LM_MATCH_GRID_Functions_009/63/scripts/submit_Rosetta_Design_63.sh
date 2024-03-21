#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_63
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/63/scripts/AI_Rosetta_Design_63.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/63/scripts/AI_Rosetta_Design_63.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/63
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/63/scripts/Rosetta_Design_63.sh
