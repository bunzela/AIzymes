"""
Manages all MPNN methods for protein design

Functions:
    prepare_ProteinMPNN():  Prepares commands for ProteinMPNN job submission.
    prepare_SolubleMPNN():  Prepares commands for SolubleMPNN job submission.
    prepare_LigandMPNN():   Prepares commands for LigandMPNN job submission.
    make_bias_dict():       Creates and writes a bias dictionary for MPNN based on an input PDB.
    ProteinMPNN_check():    Checks that ProteinMPNN is installed.
    LigandMPNN_check():     Checks that LigandMPNN is installed.

Modules Required:
    helper_002
"""
import os
import logging
import json
import numpy as np
import sys

from helper_002               import sequence_from_pdb

def prepare_LigandMPNN(self, 
                       index, 
                       cmd,
                       gpu_id = None):
    """
    Uses LigandMPNN to redesign the input structure with index.

    Parameters:
        index (str):    The index of the designed variant.
        cmd (str):      Growing list of commands to be exected by run_design using submit_job.
    Optional Parameters:
        gpu_id (int):   ID of the GPU to be used.
    Returns:
        cmd (str):      Command to be exected by run_design using submit_job
    """
    
    LigandMPNN_check(self)
        
    folder_ligandmpnn = f"{self.FOLDER_DESIGN}/{index}/LigandMPNN"
    os.makedirs(folder_ligandmpnn, exist_ok=True)
    
    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"]
    if "parent" in PDB_input:
        seq_input = PDB_input
    else:
        seq_input = "_".join(PDB_input.split("_")[:-2]+PDB_input.split("_")[-1:])
        
    make_bias_dict(self, PDB_input, folder_ligandmpnn)

    cmd = f"""### LigandMPNN ###
"""

    if  gpu_id != None:
        cmd += f"""
export CUDA_VISIBLE_DEVICES={gpu_id}
"""
        
    cmd += f"""    
# Copy input PDB
cp {PDB_input}.pdb {folder_ligandmpnn}

echo 'XXXXXXXXXXXXXXXXXXXXXX Extract catalytic residue information -- TO DO!!!! FIX RESIDUES!!!'

# Extract catalytic residue information -- TO DO!!!! FIX RESIDUES!!!
# cat_resi = int(all_scores_df.at[parent_index, 'cat_resi'])
fixed_residues = f"A1"

echo 'XXXXXXXXXXXXXXXXXXXXXX ADD --use_soluble_model TAG???'
echo 'XXXXXXXXXXXXXXXXXXXXXX ADD --bias_by_res_jsonl {os.path.join(folder_solublempnn, 'bias_by_res.jsonl')}'

# Run LigandMPNN
{self.bash_args} python {os.path.join(self.FOLDER_LigandMPNN, 'run.py')} \
--model_type ligand_mpnn \
--temperature {self.LigandMPNN_T} \
--seed 37 \
--pdb_path os.path.join(ligand_mpnn_folder, f"{WT}_Rosetta_Relax_{parent_index}.pdb"),
--out_folder {folder_ligandmpnn} \
--pack_side_chains 1 \
--number_of_packs_per_design 100 \
--fixed_residues fixed_residues

# Find highest scoring sequence
{self.bash_args} python {os.path.join(self.FOLDER_PARENT, 'find_highest_scoring_sequence.py')} \
--sequence_wildcard {self.FOLDER_HOME}/{self.WT}_with_X_as_wildecard.seq \
--sequence_parent   {seq_input}.seq \
--sequence_in       {folder_ligandmpnn}/seqs/{os.path.splitext(os.path.basename(PDB_input))[0]}.fa \
--sequence_out      {self.FOLDER_DESIGN}/{index}/{self.WT}_{index}.seq 
"""                              
    
    if self.REMOVE_TMP_FILES:
        cmd += f"""
# Removing temporary directory
rm -r LigandMPNN
"""
        
    return(cmd)
    
def prepare_ProteinMPNN(self,
                        index, 
                        cmd,
                        gpu_id = None):
    """
    Uses ProteinMPNN to redesign the input structure with index.

    Args:
        index (str): The index of the designed variant.
        cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
        cmd (str): Command to be exected by run_design using submit_job
    """
    
    ProteinMPNN_check(self)
        
    folder_proteinmpnn = f"{self.FOLDER_DESIGN}/{index}/ProteinMPNN"
    os.makedirs(folder_proteinmpnn, exist_ok=True)
    
    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"]
    if "parent" in PDB_input:
        seq_input = PDB_input
    else:
        seq_input = "_".join(PDB_input.split("_")[:-2]+PDB_input.split("_")[-1:])
    
    make_bias_dict(self, PDB_input, folder_proteinmpnn)

    cmd = f"""### ProteinMPNN ###
"""

    if  gpu_id != None:
        cmd += f"""
export CUDA_VISIBLE_DEVICES={gpu_id}
"""
        
    cmd += f"""    
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
--sequence_out      {self.FOLDER_DESIGN}/{index}/{self.WT}_{index}.seq 

"""                              
    
    return(cmd)

def prepare_SolubleMPNN(self, 
                        index,
                        cmd,
                        gpu_id = None):
    """
    Uses SolubleMPNN to redesign the input structure with index.

    Args:
        index (str): The index of the designed variant.
        cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
        cmd (str): Command to be exected by run_design using submit_job
    """
    
    ProteinMPNN_check(self)
        
    folder_solublempnn = f"{self.FOLDER_DESIGN}/{index}/SolubleMPNN"
    os.makedirs(folder_solublempnn, exist_ok=True)
    
    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"]
    if "parent" in PDB_input:
        seq_input = PDB_input
    else:
        seq_input = "_".join(PDB_input.split("_")[:-2]+PDB_input.split("_")[-1:])
        
    make_bias_dict(self, PDB_input, folder_solublempnn)

    cmd = f"""### SolubleMPNN ###
"""

    if  gpu_id != None:
        cmd += f"""
export CUDA_VISIBLE_DEVICES={gpu_id}
"""
        
    cmd += f"""    
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
--sequence_in       {folder_solublempnn}/seqs/{os.path.basename(PDB_input)}.fa \
--sequence_out      {self.FOLDER_DESIGN}/{index}/{self.WT}_{index}.seq 

"""                              
    
    return(cmd)
    
def make_bias_dict(self, PDB_input, folder_mpnn):
    """
    Creates a bias dictionary for the input PDB and writes it to a JSON file.
    
    Args:
        PDB_input (str): Path to the input PDB file.
        folder_mpnn (str): Folder where the bias JSON will be saved.
    """ 
    
    # Prepare input JSON for bias dictionary creation
    seq = sequence_from_pdb(PDB_input)
    input_json = {"name": f"{os.path.basename(PDB_input)}", "seq_chain_A": seq}

    # Create bias dictionary
    mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    mpnn_alphabet_dict = {aa: residue for residue, aa in enumerate(mpnn_alphabet)}
    
    bias_dict = {}
    for chain_key, sequence in input_json.items():
        if chain_key.startswith('seq_chain_'):
            chain = chain_key[-1]
            chain_length = len(sequence)
            bias_per_residue = np.zeros([chain_length, 21])  # 21 for each amino acid in the alphabet

            # Apply a positive bias for the amino acid at each position
            for idx, aa in enumerate(sequence):

                # Ensure the amino acid is in the defined alphabet
                if aa not in mpnn_alphabet_dict:  
                    logging.error(f"Non-standard amino acid {aa} at residue {idx} in make_bias_dict() / designMPNN.py.")
                    sys.exit()
                
                # Add bias to input sequence
                aa_index = mpnn_alphabet_dict[aa]
                if "Soluble" in folder_mpnn:
                    bias_per_residue[idx, aa_index] = self.SolubleMPNN_BIAS 
                elif "Protein" in folder_mpnn:
                    bias_per_residue[idx, aa_index] = self.ProteinMPNN_BIAS 
                else:
                    logging.error(f"MPNN_BIAS not defined. Error in make_bias_dict() / designMPNN.py.")
                    sys.exit()  

                # Forbidden residues
                for forbidden_aa in self.FORBIDDEN_AA:
                    aa_index = mpnn_alphabet_dict[forbidden_aa]
                    bias_per_residue[idx, aa_index] = -1e6
                    
            # Restrict amino acids to the restricted set
            if self.RESTRICT_RESIDUES is not None:
                for idx, resns in self.RESTRICT_RESIDUES:
                    for aa_index, aa in enumerate(mpnn_alphabet):
                        if aa in resns:
                            bias_per_residue[idx-1, aa_index] = 1
                        else:
                            bias_per_residue[idx-1, aa_index] = -1e6
                    
            bias_dict[input_json["name"]] = {chain: bias_per_residue.tolist()}

    # Write the bias dictionary to a JSON file
    bias_json_path = os.path.join(folder_mpnn, "bias_by_res.jsonl")
    with open(bias_json_path, 'w') as f:
        json.dump(bias_dict, f)
        f.write('\n')       
        
def ProteinMPNN_check(self):
    """
    Checks if ProteinMPNN is installed
    """
    if not os.path.exists(self.FOLDER_ProteinMPNN):
        logging.error(f"ProteinMPNN not installed in {self.FOLDER_ProteinMPNN}.")
        logging.error(f"Install using: 'cd {self.FOLDER_ProteinMPNN} && git clone https://github.com/dauparas/ProteinMPNN.git'")
        sys.exit()
        
def LigandMPNN_check(self):
    """
    Checks if LigandMPNN is installed
    """
    if not os.path.exists(self.FOLDER_LigandMPNN):
        logging.error(f"LigandMPNN not installed in {self.FOLDER_LigandMPNN}.")
        logging.error(f"Install using: 'cd {self.FOLDER_LigandMPNN} && git clone https://github.com/dauparas/LigandMPNN.git'")
        sys.exit()
    else:
        logging.error("LigandMPNN NOT WORKING YET!!! Problem with introducing biasedict!!!")
        sys.exit()
        

    