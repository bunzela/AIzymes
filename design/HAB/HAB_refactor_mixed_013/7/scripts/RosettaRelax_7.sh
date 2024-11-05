
    
cpptraj -i /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_CPPTraj_old_bb.in &>            /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_CPPTraj_old_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_CPPTraj_lig.in &>            /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_CPPTraj_lig.out
cpptraj -i /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_CPPTraj_new_bb.in &>            /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_CPPTraj_new_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_CPPTraj_aligned.in &>            /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_CPPTraj_aligned.out

# Cleanup structures
sed -i '/END/d' /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_aligned.pdb
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_lig.pdb
# Assemble structure
cat /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_aligned.pdb >> /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_input.pdb
cat /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_lig.pdb     >> /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_input.pdb
sed -i '/TER/d' /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_input.pdb

# Run Rosetta Relax
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease                 -s                                        /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/RosettaRelax/7vuu_7_input.pdb                 -extra_res_fa                             /home/bunzelh/AIzymes/design/HAB/Input/5TS.params                 -parser:protocol                          /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/scripts/RosettaRelax_7.xml                 -out:file:scorefile                       /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/score_RosettaRelax.sc                 -nstruct                                  1                 -ignore_zero_occupancy                    false                 -corrections::beta_nov16                  true                 -run:preserve_header                      true                 -overwrite 

# Rename the output file
mv /home/bunzelh/AIzymes/design/HAB/HAB_refactor_mixed_013/7/7vuu_7_input_0001.pdb 7vuu_RosettaRelax_7.pdb
sed -i '/        H  /d' 7vuu_RosettaRelax_7.pdb

