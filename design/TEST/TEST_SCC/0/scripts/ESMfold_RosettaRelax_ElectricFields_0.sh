### ESMfold ###
    
python /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/parent/ESMfold.py --sequence_file /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_0.seq --output_file   /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_ESMfold_bb.pdb 

sed -i '/PARENT N\/A/d' /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_ESMfold_bb.pdb

    
cpptraj -i /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_CPPTraj_old_bb.in &>            /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_CPPTraj_old_bb.out
cpptraj -i /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_CPPTraj_lig.in &>            /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_CPPTraj_lig.out
cpptraj -i /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_CPPTraj_new_bb.in &>            /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_CPPTraj_new_bb.out
cpptraj -i /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_CPPTraj_aligned.in &>            /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_CPPTraj_aligned.out

# Cleanup structures
sed -i '/END/d' /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_aligned.pdb
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_lig.pdb
# Assemble structure
cat /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_aligned.pdb >> /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_input.pdb
cat /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_lig.pdb     >> /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_input.pdb
sed -i '/TER/d' /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_input.pdb

cat /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ESMFold/7vuu_0_input.pdb > /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_ESMfold_0.pdb

### RosettaRelax ###
    
# Assemble structure
cat /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_ESMfold_0.pdb > /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/RosettaRelax/7vuu_0_input.pdb

# Run Rosetta Relax
/scratch-scc/projects/scc_mmtm_bunzel/rosetta.source.release-371/main/source//bin/rosetta_scripts.linuxgccrelease \
    -s                                        /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/RosettaRelax/7vuu_0_input.pdb \
    -extra_res_fa                             /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/Input/5TS.params \
    -parser:protocol                          /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/scripts/RosettaRelax_0.xml \
    -out:file:scorefile                       /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/score_RosettaRelax.sc \
    -nstruct                                  1 \
    -ignore_zero_occupancy                    false \
    -corrections::beta_nov16                  true \
    -run:preserve_header                      true \
    -overwrite 

# Rename the output file
mv /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_0_input_0001.pdb 7vuu_RosettaRelax_0.pdb
sed -i '/        H  /d' 7vuu_RosettaRelax_0.pdb

### ElectricFields ###

# Delete everything after 'CONECT' and remove hydrogens 
sed -n '/CONECT/q;p' /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_RosettaRelax_0.pdb > \
                     /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaRelax_0.pdb
sed -i '/ H /d'      /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaRelax_0.pdb

# Make AMBER files
tleap -f             /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaRelax_0_tleap.in > \
                     /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaRelax_0_tleap.out
mv leap.log          /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaRelax_0_tleap.log

# Add field calculation command
python   /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/../../../src/FieldTools.py \
-nc      /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaRelax_0.rst7 \
-parm    /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaRelax_0.parm7 \
-out     /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaRelax_0_fields.pkl \
-target  /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/parent/field_target.dat \
-solvent WAT


