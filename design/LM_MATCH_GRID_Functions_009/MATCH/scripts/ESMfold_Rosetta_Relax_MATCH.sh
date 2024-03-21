
    
python /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/ESMfold.py /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/1ohp_ESMfold_output_MATCH.pdb /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/1ohp_MATCH.seq

sed -i '/PARENT N\/A/d' /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/1ohp_ESMfold_output_MATCH.pdb
cpptraj -i /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/CPPTraj_Apo_MATCH.in           &>            /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/CPPTraj_Apo_MATCH.out
cpptraj -i /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/CPPTraj_Lig_MATCH.in           &>            /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/CPPTraj_Lig_MATCH.out
cpptraj -i /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/CPPTraj_no_hydrogens_MATCH.in  &>            /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/CPPTraj_no_hydrogens_MATCH.out
cpptraj -i /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/CPPTraj_aligned_MATCH.in       &>            /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/CPPTraj_aligned_MATCH.out

# Assemble the final protein
sed -i '/END/d' /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/1ohp_ESMfold_aligned_MATCH.pdb
# Return HETATM to ligand output and remove TER
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/1ohp_CPPTraj_Lig_MATCH.pdb

cp /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/1ohp_ESMfold_aligned_MATCH.pdb    /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/1ohp_ESMfold_MATCH_APO.pdb

# Run Rosetta Relax
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease                 -s                                        /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/1ohp_ESMfold_MATCH_APO.pdb                 -extra_res_fa                             /home/lmerlicek/AIzymes/design/Input/5TS.params                 -parser:protocol                          /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/scripts/Rosetta_Relax_MATCH.xml                 -out:file:scorefile                       /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/score_rosetta_relax.sc                 -nstruct                                  1                 -ignore_zero_occupancy                    false                 -corrections::beta_nov16                  true                 -run:preserve_header                      true                 -overwrite -ex1 -ex2

# Rename the output file
mv 1ohp_ESMfold_MATCH_APO_0001.pdb 1ohp_Rosetta_Relax_MATCH_APO.pdb
sed -i '/        H  /d' 1ohp_Rosetta_Relax_MATCH_APO.pdb

# Align relaxed ESM prediction of scaffold without hydrogens
cpptraj -i /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/1ohp_Rosetta_Relax_aligned_MATCH_APO.in           &>            /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/ESMfold/1ohp_Rosetta_Relax_aligned_MATCH_APO.out
sed -i '/END/d' /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/1ohp_Rosetta_Relax_aligned_MATCH_APO.pdb
