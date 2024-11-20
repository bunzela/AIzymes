
    
# Copy input PDB
cp /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/parent/7vuu.pdb /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN

# Parse chains
 python /home/bunzelh/ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN --output_path /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/parsed_chains.jsonl 

# Assign fixed chains
 python /home/bunzelh/ProteinMPNN/helper_scripts/assign_fixed_chains.py --input_path /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/parsed_chains.jsonl --output_path /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/assigned_chains.jsonl --chain_list A 

# Make fixed positions dict
 python /home/bunzelh/ProteinMPNN/helper_scripts/make_fixed_positions_dict.py --input_path /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/parsed_chains.jsonl --output_path /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/fixed_positions.jsonl --chain_list A --position_list '4 8 16 21 25 28 40 41 44 52 57 60 61' 

# Protein MPNN run
 python /home/bunzelh/ProteinMPNN/protein_mpnn_run.py --jsonl_path /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/parsed_chains.jsonl --chain_id_jsonl /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/assigned_chains.jsonl --fixed_positions_jsonl /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/fixed_positions.jsonl --bias_by_res_jsonl /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/bias_by_res.jsonl --out_folder /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN --num_seq_per_target 100 --sampling_temp 0.1 --seed 37 --batch_size 1

# Find highest scoring sequence
 python /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/parent/find_highest_scoring_sequence.py --sequence_wildcard /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/7vuu_with_X_as_wildecard.seq --sequence_parent   /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/parent/7vuu.seq --sequence_in       /home/bunzelh/AIzymes/design/HAB/HAB_refactor_ProtMPNN_013/5/ProteinMPNN/seqs/7vuu.fa --sequence_out      7vuu_5.seq 
