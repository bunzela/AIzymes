#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_4
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/4/scripts/AI_Rosetta_Design_4.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/4/scripts/AI_Rosetta_Design_4.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/4
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/4/scripts/Rosetta_Design_4.sh
