### ESMfold ###
    
python /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/parent/ESMfold.py --sequence_file /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/7vuu_2.seq --output_file   /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/7vuu_ESMfold_2.pdb 

sed -i '/PARENT N\/A/d' /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/7vuu_ESMfold_2.pdb
### RosettaRelax ###
    
# Cleanup structures
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_CPPTraj_old_bb.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_CPPTraj_old_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_CPPTraj_lig.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_CPPTraj_lig.out
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_CPPTraj_new_bb.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_CPPTraj_new_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_CPPTraj_aligned.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_CPPTraj_aligned.out
sed -i '/END/d' /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_aligned.pdb
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_lig.pdb
# Assemble structure
cat /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_aligned.pdb >> /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_input.pdb
cat /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_lig.pdb     >> /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_input.pdb
sed -i '/TER/d' /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_input.pdb
# Run Rosetta
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease \
                -s                                        /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/RosettaRelax/7vuu_2_input.pdb \
                -extra_res_fa                             /home/bunzelh/AIzymes/design/TEST/Input/5TS.params \
                -parser:protocol                          /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/scripts/RosettaRelax_2.xml \
                -out:file:scorefile                       /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/score_RosettaRelax.sc \
                -nstruct                                  1 \
                -ignore_zero_occupancy                    false \
                -corrections::beta_nov16                  true \
                -run:preserve_header                      true \
                -overwrite 

# Rename the output file
mv /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/7vuu_2_input_0001.pdb 7vuu_RosettaRelax_2.pdb
sed -i '/        H  /d' 7vuu_RosettaRelax_2.pdb

### ElectricFields ###

# Delete everything after 'CONECT' and remove hydrogens 
sed -n '/CONECT/q;p' /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/7vuu_RosettaRelax_2.pdb > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/ElectricFields/7vuu_RosettaRelax_2.pdb
sed -i '/ H /d'      /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/ElectricFields/7vuu_RosettaRelax_2.pdb

# Make AMBER files
tleap -f             /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/ElectricFields/7vuu_RosettaRelax_2_tleap.in > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/ElectricFields/7vuu_RosettaRelax_2_tleap.out
mv leap.log          /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/ElectricFields/7vuu_RosettaRelax_2_tleap.log

# Add field calculation command
python   /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/../../../src/FieldTools.py \
-nc      /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/ElectricFields/7vuu_RosettaRelax_2.rst7 \
-parm    /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/ElectricFields/7vuu_RosettaRelax_2.parm7 \
-out     /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/2/ElectricFields/7vuu_RosettaRelax_2_fields.pkl \
-target  /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/parent/field_target.dat \
-solvent WAT


