#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_65
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/65/scripts/AI_Rosetta_Design_65.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/65/scripts/AI_Rosetta_Design_65.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/65
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/65/scripts/Rosetta_Design_65.sh
