#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_78
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/78/scripts/AI_Rosetta_Design_78.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/78/scripts/AI_Rosetta_Design_78.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/78
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/78/scripts/Rosetta_Design_78.sh
