#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_98
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/98/scripts/AI_Rosetta_Design_98.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/98/scripts/AI_Rosetta_Design_98.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/98
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/98/scripts/Rosetta_Design_98.sh
