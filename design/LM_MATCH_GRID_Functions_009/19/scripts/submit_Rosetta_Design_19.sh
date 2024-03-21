#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_19
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/19/scripts/AI_Rosetta_Design_19.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/19/scripts/AI_Rosetta_Design_19.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/19
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/19/scripts/Rosetta_Design_19.sh
