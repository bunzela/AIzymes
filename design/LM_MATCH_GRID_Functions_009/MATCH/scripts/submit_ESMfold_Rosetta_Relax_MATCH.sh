#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_ESMfold_Rosetta_Relax_MATCH
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/scripts/AI_ESMfold_Rosetta_Relax_MATCH.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/scripts/AI_ESMfold_Rosetta_Relax_MATCH.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/scripts/ESMfold_Rosetta_Relax_MATCH.sh
