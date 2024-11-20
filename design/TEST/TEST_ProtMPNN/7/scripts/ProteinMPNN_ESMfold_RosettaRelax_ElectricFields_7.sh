### ProteinMPNN ###
    
# Copy input PDB
cp /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/parent/7vuu.pdb /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN

# Parse chains
 python /home/bunzelh/ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN --output_path /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/parsed_chains.jsonl 

# Assign fixed chains
 python /home/bunzelh/ProteinMPNN/helper_scripts/assign_fixed_chains.py --input_path /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/parsed_chains.jsonl --output_path /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/assigned_chains.jsonl --chain_list A 

# Make fixed positions dict
 python /home/bunzelh/ProteinMPNN/helper_scripts/make_fixed_positions_dict.py --input_path /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/parsed_chains.jsonl --output_path /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/fixed_positions.jsonl --chain_list A --position_list '4 8 16 21 25 28 40 41 44 52 57 60 61' 

# Protein MPNN run
 python /home/bunzelh/ProteinMPNN/protein_mpnn_run.py --jsonl_path /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/parsed_chains.jsonl --chain_id_jsonl /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/assigned_chains.jsonl --fixed_positions_jsonl /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/fixed_positions.jsonl --bias_by_res_jsonl /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/bias_by_res.jsonl --out_folder /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN --num_seq_per_target 100 --sampling_temp 0.1 --seed 37 --batch_size 1

# Find highest scoring sequence
 python /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/parent/find_highest_scoring_sequence.py --sequence_wildcard /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7vuu_with_X_as_wildecard.seq --sequence_parent   /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/parent/7vuu.seq --sequence_in       /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ProteinMPNN/seqs/7vuu.fa --sequence_out      7vuu_7.seq 

### ESMfold ###
    
python /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/parent/ESMfold.py --sequence_file /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/7vuu_7.seq --output_file   /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/7vuu_ESMfold_7.pdb 

sed -i '/PARENT N\/A/d' /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/7vuu_ESMfold_7.pdb
### RosettaRelax ###
    
# Cleanup structures
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_CPPTraj_old_bb.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_CPPTraj_old_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_CPPTraj_lig.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_CPPTraj_lig.out
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_CPPTraj_new_bb.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_CPPTraj_new_bb.out
cpptraj -i /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_CPPTraj_aligned.in &> \
           /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_CPPTraj_aligned.out
sed -i '/END/d' /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_aligned.pdb
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_lig.pdb
# Assemble structure
cat /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_aligned.pdb >> /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_input.pdb
cat /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_lig.pdb     >> /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_input.pdb
sed -i '/TER/d' /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_input.pdb
# Run Rosetta
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease \
                -s                                        /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/RosettaRelax/7vuu_7_input.pdb \
                -extra_res_fa                             /home/bunzelh/AIzymes/design/TEST/Input/5TS.params \
                -parser:protocol                          /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/scripts/RosettaRelax_7.xml \
                -out:file:scorefile                       /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/score_RosettaRelax.sc \
                -nstruct                                  1 \
                -ignore_zero_occupancy                    false \
                -corrections::beta_nov16                  true \
                -run:preserve_header                      true \
                -overwrite 

# Rename the output file
mv /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/7vuu_7_input_0001.pdb 7vuu_RosettaRelax_7.pdb
sed -i '/        H  /d' 7vuu_RosettaRelax_7.pdb

### ElectricFields ###

# Delete everything after 'CONECT' and remove hydrogens 
sed -n '/CONECT/q;p' /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/7vuu_ProteinMPNN_7.pdb > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ElectricFields/7vuu_ProteinMPNN_7.pdb
sed -i '/ H /d'      /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ElectricFields/7vuu_ProteinMPNN_7.pdb

# Make AMBER files
tleap -f             /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ElectricFields/7vuu_ProteinMPNN_7_tleap.in > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ElectricFields/7vuu_ProteinMPNN_7_tleap.out
mv leap.log          /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ElectricFields/7vuu_ProteinMPNN_7_tleap.log

# Add field calculation command
python   /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/../../../src/FieldTools.py \
-nc      /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ElectricFields/7vuu_ProteinMPNN_7.rst7 \
-parm    /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ElectricFields/7vuu_ProteinMPNN_7.parm7 \
-out     /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/7/ElectricFields/7vuu_ProteinMPNN_7_fields.pkl \
-target  /home/bunzelh/AIzymes/design/TEST/TEST_ProtMPNN/parent/field_target.dat \
-solvent WAT


