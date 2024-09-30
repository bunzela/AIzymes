import os
import logging
import shutil
import json
import numpy as np
from pandas import DataFrame  

def prepare_ProteinMPNN(parent_index, new_index, all_scores_df):
    """
    Executes the ProteinMPNN pipeline for a given protein structure and generates
    new protein sequences with potentially higher functional scores.

    Parameters:
    - parent_index (str): The index of the parent protein variant.
    - new_index (str): The index assigned to the new protein variant.
    - all_scores_df (DataFrame): A DataFrame containing information for protein variants.
    - --------------------------------GLOBAL variables used ----------------------------
    - FOLDER_HOME (str): The base directory where ProteinMPNN and related files are located.
    - WT (str): The wild type or reference protein identifier.
    - DESIGN (str): A string representing positions and types of amino acids to design with RosettaDesign.
    - ProteinMPNN_T (float): The sampling temperature for ProteinMPNN.
    - EXPLORE (bool): Flag to indicate whether exploration mode is enabled.
    - PMPNN_BIAS (float): The bias value for ProteinMPNN parent sequence retention.

    Returns:
    None: The function primarily works by side effects, finally producing the highes scoring sequence 
    in the specified directories.
    
    Note:
    This function assumes the ProteinMPNN toolkit is available and properly set up in the specified location.
    It involves multiple subprocess calls to Python scripts for processing protein structures and generating new sequences.
    """

    # Testing PMPNN_submit.py
    cmd = f'python {os.getcwd()}/PMPNN_submit.py --index {parent_index} --new_index {new_index} --home_folder {FOLDER_HOME}'
    with open(f'{FOLDER_HOME}/{new_index}/scripts/PMPNN_ESM_Relax_{new_index}.sh','w') as file: file.write(cmd)
    submit_job(index=new_index, job="PMPNN_ESM_Relax")
    return

    # Ensure ProteinMPNN is available
    if not os.path.exists(f'{FOLDER_HOME}/../ProteinMPNN'):
        logging.error(f"ProteinMPNN not installed in {FOLDER_HOME}/../ProteinMPNN.")
        logging.error("Install using: git clone https://github.com/dauparas/ProteinMPNN.git")
        return

    # Prepare file paths
    pdb_file = f"{FOLDER_HOME}/{parent_index}/{WT}_Rosetta_Relax_{parent_index}.pdb"
    if not os.path.isfile(pdb_file):
        logging.error(f"{pdb_file} not present!")
        return

    protein_mpnn_folder = f"{FOLDER_HOME}/{new_index}/ProteinMPNN"
    os.makedirs(protein_mpnn_folder, exist_ok=True)
    shutil.copy(pdb_file, os.path.join(protein_mpnn_folder, f"{WT}_Rosetta_Relax_{parent_index}.pdb"))

    seq = extract_sequence_from_pdb(pdb_file)
    with open(os.path.join(protein_mpnn_folder, f"Rosetta_Relax_{parent_index}.seq"), "w") as f:
        f.write(seq)
    

    # Run ProteinMPNN steps using subprocess after creating the bias file
    helper_scripts_path = f"{FOLDER_HOME}/../ProteinMPNN/helper_scripts"
    protein_mpnn_path = f"{FOLDER_HOME}/../ProteinMPNN"

    # Prepare input JSON for bias dictionary creation
    input_json = {"name": f"{WT}_Rosetta_Relax_{parent_index}", "seq_chain_A": seq}

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
                    bias_per_residue[idx, aa_index] = PMPNN_BIAS  # Use the global bias variable

            bias_dict[input_json["name"]] = {chain: bias_per_residue.tolist()}

    # Write the bias dictionary to a JSON file
    bias_json_path = os.path.join(protein_mpnn_folder, "bias_by_res.jsonl")
    with open(bias_json_path, 'w') as f:
        json.dump(bias_dict, f)
        f.write('\n')

    # Parse multiple chains
    run_command([
        "python", os.path.join(helper_scripts_path, "parse_multiple_chains.py"),
        "--input_path", protein_mpnn_folder,
        "--output_path", os.path.join(protein_mpnn_folder, "parsed_chains.jsonl")
    ])

    # Assign fixed chains
    run_command([
        "python", os.path.join(helper_scripts_path, "assign_fixed_chains.py"),
        "--input_path", os.path.join(protein_mpnn_folder, "parsed_chains.jsonl"),
        "--output_path", os.path.join(protein_mpnn_folder, "assigned_chains.jsonl"),
        "--chain_list", 'A'
    ])

    # Make fixed positions dict
    run_command([
        "python", os.path.join(helper_scripts_path, "make_fixed_positions_dict.py"),
        "--input_path", os.path.join(protein_mpnn_folder, "parsed_chains.jsonl"),
        "--output_path", os.path.join(protein_mpnn_folder, "fixed_positions.jsonl"),
        "--chain_list", 'A',
        "--position_list", " ".join(DESIGN.split(","))
    ])

    # Protein MPNN run
    run_command([
        "python", os.path.join(protein_mpnn_path, "protein_mpnn_run.py"),
        "--jsonl_path", os.path.join(protein_mpnn_folder, "parsed_chains.jsonl"),
        "--chain_id_jsonl", os.path.join(protein_mpnn_folder, "assigned_chains.jsonl"),
        "--fixed_positions_jsonl", os.path.join(protein_mpnn_folder, "fixed_positions.jsonl"),
        "--bias_by_res_jsonl", os.path.join(protein_mpnn_folder, "bias_by_res.jsonl"),
        "--out_folder", protein_mpnn_folder,
        "--num_seq_per_target", "100",
        "--sampling_temp", ProteinMPNN_T,
        "--seed", "37",
        "--batch_size", "1"
    ])
    

    # Find highest scoring sequence
    highest_scoring_sequence = find_highest_scoring_sequence(protein_mpnn_folder, parent_index, input_sequence_path=
                                                             f"{FOLDER_HOME}/input_sequence_with_X_as_wildecard.seq")

    # Save highest scoring sequence and prepare for ESMfold
    with open(os.path.join(protein_mpnn_folder, f"{WT}_{new_index}.seq"), "w") as f:
        f.write(highest_scoring_sequence)
    
    if highest_scoring_sequence:
        logging.info(f"Ran ProteinMPNN for index {parent_index} and found a new sequence with index {new_index}.")
    else:
        logging.error(f"Failed to find a new sequnce for index {parent_index} with ProteinMPNN.")
    
    all_scores_df = save_cat_res_into_all_scores_df(all_scores_df, new_index, pdb_file, from_parent_struct=False)
        
    # Run ESMfold Relax with the ProteinMPNN Flag
    run_ESMfold_RosettaRelax(index=new_index, all_scores_df=all_scores_df, OnlyRelax=False, \
                             ProteinMPNN=True, ProteinMPNN_parent_index=parent_index, EXPLORE=EXPLORE)