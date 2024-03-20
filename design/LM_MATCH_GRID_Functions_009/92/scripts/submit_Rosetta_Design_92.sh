#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_92
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/92/scripts/AI_Rosetta_Design_92.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/92/scripts/AI_Rosetta_Design_92.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/92
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/92/scripts/Rosetta_Design_92.sh
