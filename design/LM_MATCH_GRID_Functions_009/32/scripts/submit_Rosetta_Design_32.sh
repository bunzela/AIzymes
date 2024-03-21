#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_32
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/32/scripts/AI_Rosetta_Design_32.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/32/scripts/AI_Rosetta_Design_32.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/32
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/32/scripts/Rosetta_Design_32.sh
