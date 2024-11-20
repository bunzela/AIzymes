
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease    -s                                        /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32/1ohp_RosettaRelax_32.pdb     -in:file:native                           /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/32/1ohp_RosettaRelax_32.pdb     -run:preserve_header                      true     -extra_res_fa                             /home/bunzelh/AIzymes/design/VD/Input/5TS.params     -enzdes:cstfile                           /home/bunzelh/AIzymes/design/VD/Input/5TS_enzdes_planar_tAB100.cst     -enzdes:cst_opt                           true     -parser:protocol                          /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/83/scripts/RosettaDesign_83.xml     -out:file:scorefile                       /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/83/score_RosettaDesign.sc     -nstruct                                  1      -ignore_zero_occupancy                    false      -corrections::beta_nov16                  true     -overwrite -ex1 -ex2
        
mv 1ohp_RosettaRelax_32_0001.pdb 1ohp_RosettaDesign_83.pdb 

 python /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/parent/extract_sequence_from_pdb.py --pdb_in 1ohp_RosettaDesign_83.pdb --sequence_out 1ohp_83.seq

