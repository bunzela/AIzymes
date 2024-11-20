
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease    -s                                        /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/parent/7vuu.pdb     -in:file:native                           /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/parent/7vuu.pdb     -run:preserve_header                      true     -extra_res_fa                             /home/bunzelh/AIzymes/design/HAB/Input/5TS.params     -enzdes:cstfile                           /home/bunzelh/AIzymes/design/HAB/Input/5TS_enzdes_planar_tAB100.cst     -enzdes:cst_opt                           true     -parser:protocol                          /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2/scripts/RosettaDesign_2.xml     -out:file:scorefile                       /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/2/score_RosettaDesign.sc     -nstruct                                  1      -ignore_zero_occupancy                    false      -corrections::beta_nov16                  true     -overwrite 
        
mv 7vuu_0001.pdb 7vuu_RosettaDesign_2.pdb 

 python /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/parent/extract_sequence_from_pdb.py --pdb_in 7vuu_RosettaDesign_2.pdb --sequence_out 7vuu_2.seq

