#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_31
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/31/scripts/AI_Rosetta_Design_31.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/31/scripts/AI_Rosetta_Design_31.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/31
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/31/scripts/Rosetta_Design_31.sh
