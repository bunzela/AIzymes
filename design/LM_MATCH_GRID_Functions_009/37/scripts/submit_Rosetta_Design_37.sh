#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_37
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/37/scripts/AI_Rosetta_Design_37.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/37/scripts/AI_Rosetta_Design_37.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/37
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/37/scripts/Rosetta_Design_37.sh
