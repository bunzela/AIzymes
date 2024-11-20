
    
cpptraj -i /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_CPPTraj_old_bb.in &>            /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_CPPTraj_old_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_CPPTraj_lig.in &>            /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_CPPTraj_lig.out
cpptraj -i /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_CPPTraj_new_bb.in &>            /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_CPPTraj_new_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_CPPTraj_aligned.in &>            /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_CPPTraj_aligned.out

# Cleanup structures
sed -i '/END/d' /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_aligned.pdb
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_lig.pdb
# Assemble structure
cat /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_aligned.pdb >> /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_input.pdb
cat /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_lig.pdb     >> /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_input.pdb
sed -i '/TER/d' /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_input.pdb

# Run Rosetta Relax
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease                 -s                                        /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/RosettaRelax/1ohp_36_input.pdb                 -extra_res_fa                             /home/bunzelh/AIzymes/design/VD/Input/5TS.params                 -parser:protocol                          /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/scripts/RosettaRelax_36.xml                 -out:file:scorefile                       /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/score_RosettaRelax.sc                 -nstruct                                  1                 -ignore_zero_occupancy                    false                 -corrections::beta_nov16                  true                 -run:preserve_header                      true                 -overwrite -ex1 -ex2

# Rename the output file
mv /home/bunzelh/AIzymes/design/VD/241016_VD_KSI_pos/36/1ohp_36_input_0001.pdb 1ohp_RosettaRelax_36.pdb
sed -i '/        H  /d' 1ohp_RosettaRelax_36.pdb

