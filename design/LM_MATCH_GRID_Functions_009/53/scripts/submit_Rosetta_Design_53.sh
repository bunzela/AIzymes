#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_53
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/53/scripts/AI_Rosetta_Design_53.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/53/scripts/AI_Rosetta_Design_53.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/53
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/53/scripts/Rosetta_Design_53.sh
