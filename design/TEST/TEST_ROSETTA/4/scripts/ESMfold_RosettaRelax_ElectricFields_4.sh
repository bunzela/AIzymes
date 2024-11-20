### ESMfold ###
    
python /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/parent/ESMfold.py --sequence_file /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/7vuu_4.seq --output_file   /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/7vuu_ESMfold_4.pdb 

sed -i '/PARENT N\/A/d' /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/7vuu_ESMfold_4.pdb
### RosettaRelax ###
    
# Cleanup structures
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_CPPTraj_old_bb.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_CPPTraj_old_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_CPPTraj_lig.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_CPPTraj_lig.out
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_CPPTraj_new_bb.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_CPPTraj_new_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_CPPTraj_aligned.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_CPPTraj_aligned.out
sed -i '/END/d' /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_aligned.pdb
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_lig.pdb
# Assemble structure
cat /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_aligned.pdb >> /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_input.pdb
cat /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_lig.pdb     >> /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_input.pdb
sed -i '/TER/d' /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_input.pdb
# Run Rosetta
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease \
                -s                                        /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/RosettaRelax/7vuu_4_input.pdb \
                -extra_res_fa                             /home/bunzelh/AIzymes/design/TEST/Input/5TS.params \
                -parser:protocol                          /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/scripts/RosettaRelax_4.xml \
                -out:file:scorefile                       /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/score_RosettaRelax.sc \
                -nstruct                                  1 \
                -ignore_zero_occupancy                    false \
                -corrections::beta_nov16                  true \
                -run:preserve_header                      true \
                -overwrite 

# Rename the output file
mv /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/7vuu_4_input_0001.pdb 7vuu_RosettaRelax_4.pdb
sed -i '/        H  /d' 7vuu_RosettaRelax_4.pdb

### ElectricFields ###

# Delete everything after 'CONECT' and remove hydrogens 
sed -n '/CONECT/q;p' /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/7vuu_RosettaRelax_4.pdb > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/ElectricFields/7vuu_RosettaRelax_4.pdb
sed -i '/ H /d'      /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/ElectricFields/7vuu_RosettaRelax_4.pdb

# Make AMBER files
tleap -f             /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/ElectricFields/7vuu_RosettaRelax_4_tleap.in > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/ElectricFields/7vuu_RosettaRelax_4_tleap.out
mv leap.log          /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/ElectricFields/7vuu_RosettaRelax_4_tleap.log

# Add field calculation command
python   /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/../../../src/FieldTools.py \
-nc      /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/ElectricFields/7vuu_RosettaRelax_4.rst7 \
-parm    /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/ElectricFields/7vuu_RosettaRelax_4.parm7 \
-out     /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/ElectricFields/7vuu_RosettaRelax_4_fields.pkl \
-target  /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/parent/field_target.dat \
-solvent WAT


