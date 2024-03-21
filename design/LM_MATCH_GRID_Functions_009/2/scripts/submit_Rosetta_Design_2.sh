#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_2
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/2/scripts/AI_Rosetta_Design_2.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/2/scripts/AI_Rosetta_Design_2.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/2
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/2/scripts/Rosetta_Design_2.sh
