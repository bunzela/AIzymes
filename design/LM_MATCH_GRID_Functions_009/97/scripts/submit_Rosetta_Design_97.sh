#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_97
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/97/scripts/AI_Rosetta_Design_97.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/97/scripts/AI_Rosetta_Design_97.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/97
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/97/scripts/Rosetta_Design_97.sh
