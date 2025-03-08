
"""
Manages all MPNN methods for protein design

This function assumes the ProteinMPNN toolkit is available and properly set up in the specified location.
It involves multiple subprocess calls to Python scripts for processing protein structures and generating new sequences.

Functions:
    prepare_ProteinMPNN: Sets up commands for ProteinMPNN job submission.

Modules Required:
    helper_001
"""
import os
import logging
import shutil
import json
import numpy as np
from pandas import DataFrame  
import re

from helper_002               import *
    
def prepare_ProteinMPNN(self, index, cmd):
    """
    Uses ProteinMPNN to redesign the input structure with index.

    Args:
    index (str): The index of the designed variant.
    cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
    cmd (str): Command to be exected by run_design using submit_job
    """
    
    ProteinMPNN_check(self)
        
    folder_proteinmpnn = f"{self.FOLDER_HOME}/{index}/ProteinMPNN"
    os.makedirs(folder_proteinmpnn, exist_ok=True)
    
    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"]
    if "parent" in PDB_input:
        seq_input = PDB_input
    else:
        seq_input = "_".join(PDB_input.split("_")[:-2]+PDB_input.split("_")[-1:])
    
    make_bias_dict(self, PDB_input, folder_proteinmpnn)

    cmd = f"""### ProteinMPNN ###
    
# Copy input PDB
cp {PDB_input}.pdb {folder_proteinmpnn}

# Parse chains
{self.bash_args} python {os.path.join(self.FOLDER_ProteinMPNN_h, 'parse_multiple_chains.py')} \
--input_path {folder_proteinmpnn} \
--output_path {os.path.join(folder_proteinmpnn, 'parsed_chains.jsonl')} 

# Assign fixed chains
{self.bash_args} python {os.path.join(self.FOLDER_ProteinMPNN_h, 'assign_fixed_chains.py')} \
--input_path {os.path.join(folder_proteinmpnn, 'parsed_chains.jsonl')} \
--output_path {os.path.join(folder_proteinmpnn, 'assigned_chains.jsonl')} \
--chain_list A 

# Make fixed positions dict
{self.bash_args} python {os.path.join(self.FOLDER_ProteinMPNN_h, 'make_fixed_positions_dict.py')} \
--input_path {os.path.join(folder_proteinmpnn, 'parsed_chains.jsonl')} \
--output_path {os.path.join(folder_proteinmpnn, 'fixed_positions.jsonl')} \
--chain_list A \
--position_list '{" ".join(self.DESIGN.split(","))}' 

# Protein MPNN run
{self.bash_args} python {os.path.join(self.FOLDER_ProteinMPNN, 'protein_mpnn_run.py')} \
--jsonl_path {os.path.join(folder_proteinmpnn, 'parsed_chains.jsonl')} \
--chain_id_jsonl {os.path.join(folder_proteinmpnn, 'assigned_chains.jsonl')} \
--fixed_positions_jsonl {os.path.join(folder_proteinmpnn, 'fixed_positions.jsonl')} \
--bias_by_res_jsonl {os.path.join(folder_proteinmpnn, 'bias_by_res.jsonl')} \
--out_folder {folder_proteinmpnn} \
--num_seq_per_target 100 \
--sampling_temp {self.ProteinMPNN_T} \
--seed 37 \
--batch_size 1

# Find highest scoring sequence
{self.bash_args} python {os.path.join(self.FOLDER_PARENT, 'find_highest_scoring_sequence.py')} \
--sequence_wildcard {self.FOLDER_HOME}/{self.WT}_with_X_as_wildecard.seq \
--sequence_parent   {seq_input}.seq \
--sequence_in       {folder_proteinmpnn}/seqs/{os.path.splitext(os.path.basename(PDB_input))[0]}.fa \
--sequence_out      {self.WT}_{index}.seq 

"""                              
    
    return(cmd)

def prepare_SolubleMPNN(self, index, cmd):
    """
    Uses SolubleMPNN to redesign the input structure with index.

    Args:
    index (str): The index of the designed variant.
    cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
    cmd (str): Command to be exected by run_design using submit_job
    """
    
    ProteinMPNN_check(self)
        
    folder_solublempnn = f"{self.FOLDER_HOME}/{index}/SolubleMPNN"
    os.makedirs(folder_solublempnn, exist_ok=True)
    
    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"]
    if "parent" in PDB_input:
        seq_input = PDB_input
    else:
        seq_input = "_".join(PDB_input.split("_")[:-2]+PDB_input.split("_")[-1:])
        
    make_bias_dict(self, PDB_input, folder_solublempnn)

    cmd = f"""### SolubleMPNN ###
    
# Copy input PDB
cp {PDB_input}.pdb {folder_solublempnn}

# Parse chains
{self.bash_args} python {os.path.join(self.FOLDER_ProteinMPNN_h, 'parse_multiple_chains.py')} \
--input_path {folder_solublempnn} \
--output_path {os.path.join(folder_solublempnn, 'parsed_chains.jsonl')} 

# Assign fixed chains
{self.bash_args} python {os.path.join(self.FOLDER_ProteinMPNN_h, 'assign_fixed_chains.py')} \
--input_path {os.path.join(folder_solublempnn, 'parsed_chains.jsonl')} \
--output_path {os.path.join(folder_solublempnn, 'assigned_chains.jsonl')} \
--chain_list A 

# Make fixed positions dict
{self.bash_args} python {os.path.join(self.FOLDER_ProteinMPNN_h, 'make_fixed_positions_dict.py')} \
--input_path {os.path.join(folder_solublempnn, 'parsed_chains.jsonl')} \
--output_path {os.path.join(folder_solublempnn, 'fixed_positions.jsonl')} \
--chain_list A \
--position_list '{" ".join(self.DESIGN.split(","))}' 

# SolubleMPNN run
{self.bash_args} python {os.path.join(self.FOLDER_ProteinMPNN, 'protein_mpnn_run.py')} \
--use_soluble_model \
--jsonl_path {os.path.join(folder_solublempnn, 'parsed_chains.jsonl')} \
--chain_id_jsonl {os.path.join(folder_solublempnn, 'assigned_chains.jsonl')} \
--fixed_positions_jsonl {os.path.join(folder_solublempnn, 'fixed_positions.jsonl')} \
--bias_by_res_jsonl {os.path.join(folder_solublempnn, 'bias_by_res.jsonl')} \
--out_folder {folder_solublempnn} \
--num_seq_per_target 100 \
--sampling_temp {self.SolubleMPNN_T} \
--seed 37 \
--batch_size 1

# Find highest scoring sequence
{self.bash_args} python {os.path.join(self.FOLDER_PARENT, 'find_highest_scoring_sequence.py')} \
--sequence_wildcard {self.FOLDER_HOME}/{self.WT}_with_X_as_wildecard.seq \
--sequence_parent   {seq_input}.seq \
--sequence_in       {folder_solublempnn}/seqs/{os.path.splitext(os.path.basename(PDB_input))[0]}.fa \
--sequence_out      {self.WT}_{index}.seq 

"""                              
    
    return(cmd)
    
def make_bias_dict(self, PDB_input, folder_mpnn):
    
    # Prepare input JSON for bias dictionary creation
    seq = sequence_from_pdb(PDB_input)
    input_json = {"name": f"{os.path.basename(PDB_input)}", "seq_chain_A": seq}

    # Create bias dictionary
    mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    mpnn_alphabet_dict = {aa: idx for idx, aa in enumerate(mpnn_alphabet)}
    
    bias_dict = {}
    for chain_key, sequence in input_json.items():
        if chain_key.startswith('seq_chain_'):
            chain = chain_key[-1]
            chain_length = len(sequence)
            bias_per_residue = np.zeros([chain_length, 21])  # 21 for each amino acid in the alphabet

            # Apply a positive bias for the amino acid at each position
            for idx, aa in enumerate(sequence):
                if aa in mpnn_alphabet_dict:  # Ensure the amino acid is in the defined alphabet
                    aa_index = mpnn_alphabet_dict[aa]
                    if "Soluble" in folder_mpnn:
                        bias_per_residue[idx, aa_index] = self.SolubleMPNN_BIAS 
                    elif "Protein" in folder_mpnn:
                        bias_per_residue[idx, aa_index] = self.ProteinMPNN_BIAS 
                    else:
                        logging.error(f"MPNN method not recognized make_bias_dict(), designMPNN.py.")
                        sys.exit()
                        
            bias_dict[input_json["name"]] = {chain: bias_per_residue.tolist()}

    # Write the bias dictionary to a JSON file
    bias_json_path = os.path.join(folder_mpnn, "bias_by_res.jsonl")
    with open(bias_json_path, 'w') as f:
        json.dump(bias_dict, f)
        f.write('\n')       
        
def ProteinMPNN_check(self):
      
    # Ensure ProteinMPNN is available
    if not os.path.exists(self.FOLDER_ProteinMPNN):
        logging.error(f"ProteinMPNN not installed in {self.FOLDER_ProteinMPNN}.")
        logging.error("Install using: git clone https://github.com/dauparas/ProteinMPNN.git")
        return
    
