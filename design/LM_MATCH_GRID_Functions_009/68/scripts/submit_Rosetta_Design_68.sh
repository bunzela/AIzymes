#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_68
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/68/scripts/AI_Rosetta_Design_68.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/68/scripts/AI_Rosetta_Design_68.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/68
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/68/scripts/Rosetta_Design_68.sh
