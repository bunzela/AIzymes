       
  
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH

echo C9 > 5TS.central
echo 14 18 26 30 55 65 80 82 99 101 112 > 5TS.pos

/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/gen_lig_grids.linuxgccrelease     -s                      1ohp_Rosetta_Relax_aligned_MATCH_APO.pdb ESMfold/1ohp_CPPTraj_Lig_MATCH.pdb     -extra_res_fa           /home/lmerlicek/AIzymes/design/Input/5TS.params     -grid_delta             0.5     -grid_lig_cutoff        5.0     -grid_bb_cutoff         2.25     -grid_active_res_cutoff 15.0     -overwrite 

mv 1ohp_Rosetta_Relax_aligned_MATCH_APO.pdb_0.gridlig 1ohp.gridlig
rm 1ohp_Rosetta_Relax_aligned_MATCH_APO.pdb_0.pos 2>1

rm -r matches
mkdir matches
cd matches

/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/match.linuxgccrelease     -s                                        ../1ohp_Rosetta_Relax_aligned_MATCH_APO.pdb     -match:lig_name                           5TS     -extra_res_fa                             /home/lmerlicek/AIzymes/design/Input/5TS.params     -match:geometric_constraint_file          /home/lmerlicek/AIzymes/design/Input/5TS_enzdes.cst     -match::scaffold_active_site_residues     ../5TS.pos     -match:required_active_site_atom_names    ../5TS.central     -match:active_site_definition_by_gridlig  ../1ohp.gridlig      -match:grid_boundary                      ../1ohp.gridlig      -gridligpath                              ../1ohp.gridlig      -overwrite      -output_format PDB      -output_matches_per_group 1      -consolidate_matches true 
