#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_RosettaMatch_MATCH
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/scripts/AI_RosettaMatch_MATCH.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/scripts/AI_RosettaMatch_MATCH.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/scripts/RosettaMatch_MATCH.sh
