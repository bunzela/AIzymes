/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease    -s                                        /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/matches/UM_4_D18_22_1ohp_Rosetta_Relax_aligned_MATCH_APO_5TS_enzdes_1.pdb     -in:file:native                           /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/MATCH/matches/UM_4_D18_22_1ohp_Rosetta_Relax_aligned_MATCH_APO_5TS_enzdes_1.pdb     -run:preserve_header                      true     -extra_res_fa                             /home/lmerlicek/AIzymes/design/Input/5TS.params     -parser:protocol                          /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/9/scripts/Rosetta_Design_9.xml     -out:file:scorefile                       /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/9/score_rosetta_design.sc     -nstruct                                  1      -ignore_zero_occupancy                    false      -corrections::beta_nov16                  true     -overwrite -ex1 -ex2
    
mv UM_4_D18_22_1ohp_Rosetta_Relax_aligned_MATCH_APO_5TS_enzdes_1_0001.pdb 1ohp_Rosetta_Design_9.pdb 
