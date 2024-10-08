import os
import subprocess
import logging
import re
import json
import shutil
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO

def run_command(command, cwd=None, capture_output=False):
    """Wrapper to execute .py files in runtime with arguments, and print error messages if they occur.
    
    Parameters:
    - command: The command to run as a list of strings.
    - cwd: Optional; The directory to execute the command in.
    - capture_output: Optional; If True, capture stdout and stderr. Defaults to False (This is to conserve memory).
    """
    try:
        # If capture_output is True, capture stdout and stderr
        if capture_output:
            result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd)
        else:
            # If capture_output is False, suppress all output by redirecting to os.devnull
            with open(os.devnull, 'w') as fnull:
                result = subprocess.run(command, stdout=fnull, stderr=fnull, text=True, check=True, cwd=cwd)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{e.cmd}' failed with return code {e.returncode}")
        logging.error(e.stderr)
        raise
    except Exception as e:
        logging.error(f"An error occurred while running command: {command}")
        raise

def submit_job(index, job, bash=False, ram=16, variables_json=None):

    if variables_json:
        FOLDER_HOME = variables_json['FOLDER_HOME']
        SUBMIT_PREFIX = variables_json['SUBMIT_PREFIX']
        ABBIE_LOCAL = variables_json['ABBIE_LOCAL']
        BLUEPEBBLE = variables_json['BLUEPEBBLE']
        BACKGROUND_JOB = variables_json['BACKGROUND_JOB']
        GRID = variables_json['GRID']

    if GRID:
        submission_script = f"""#!/bin/bash
#$ -V
#$ -cwd
#$ -N {SUBMIT_PREFIX}_{job}_{index}
#$ -hard -l mf={ram}G
#$ -o {FOLDER_HOME}/{index}/scripts/AI_{job}_{index}.out
#$ -e {FOLDER_HOME}/{index}/scripts/AI_{job}_{index}.err
"""
    if BLUEPEBBLE:
        submission_script = f"""#!/bin/bash
#SBATCH --account={BLUEPEBBLE_ACCOUNT}
#SBATCH --partition=short
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00    
#SBATCH --nodes=1          
#SBATCH --job-name={SUBMIT_PREFIX}_{job}_{index}
#SBATCH --output={FOLDER_HOME}/{index}/scripts/AI_{job}_{index}.out
#SBATCH --error={FOLDER_HOME}/{index}/scripts/AI_{job}_{index}.err
"""
        
    if BACKGROUND_JOB:
        if not os.path.isfile(f'{FOLDER_HOME}/n_running_jobs.dat'):
            with open(f'{FOLDER_HOME}/n_running_jobs.dat', 'w') as f: f.write('0')
        with open(f'{FOLDER_HOME}/n_running_jobs.dat', 'r'): jobs = int(f.read())
        with open(f'{FOLDER_HOME}/n_running_jobs.dat', 'w'): f.write(jobs+1)
        submission_script = ""

    if ABBIE_LOCAL:
        submission_script = ""
        
    submission_script += f"""
# Output folder
cd {FOLDER_HOME}/{index}
pwd
bash {FOLDER_HOME}/{index}/scripts/{job}_{index}.sh
""" 
    if BACKGROUND_JOB:
        submission_script = f"""
jobs=$(cat {FOLDER_HOME}/n_running_jobs.dat)
jobs=$((jobs - 1))
echo "$jobs" > {FOLDER_HOME}/n_running_jobs.dat
"""


    # Create the submission_script
    with open(f'{FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh', 'w') as file: file.write(submission_script)
    
    if bash:
        #Bash the submission_script for testing
        subprocess.run(f'bash {FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh', shell=True, text=True)
    else:
        #Submit the submission_script
        if GRID:
            if "ESM" in job:
                
                output = subprocess.check_output(
    (f'qsub -l h="!bs-dsvr64&!bs-dsvr58&!bs-dsvr42&'
     f'!bs-grid64&!bs-grid65&!bs-grid66&!bs-grid67&'
     f'!bs-grid68&!bs-grid69&!bs-grid70&!bs-grid71&'
     f'!bs-grid72&!bs-grid73&!bs-grid74&!bs-grid75&'
     f'!bs-grid76&!bs-grid77&!bs-grid78&!bs-headnode04&'
     f'!bs-stellcontrol05&!bs-stellsubmit05" -q regular.q '
     f'{FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh'),
    shell=True, text=True
    )
            else:
                output = subprocess.check_output(f'qsub -q regular.q \
                                                {FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh', \
                                                shell=True, text=True)
            logging.debug(output[:-1]) #remove newline at end of output
            
        if BLUEPEBBLE:
            output = subprocess.check_output(f'sbatch {FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh', \
                                             shell=True, text=True)
            logging.debug(output[:-1]) #remove newline at end of output
            
        if BACKGROUND_JOB:

            stdout_log_file_path = f'{FOLDER_HOME}/{index}/scripts/submit_{job}_{index}_stdout.log'
            stderr_log_file_path = f'{FOLDER_HOME}/{index}/scripts/submit_{job}_{index}_stderr.log'

            with open(stdout_log_file_path, 'w') as stdout_log_file, open(stderr_log_file_path, 'w') as stderr_log_file:
                process = subprocess.Popen(f'bash {FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh &', 
                                           shell=True, stdout=stdout_log_file, stderr=stderr_log_file)
        
        if ABBIE_LOCAL:

            stdout_log_file_path = f'{FOLDER_HOME}/{index}/scripts/submit_{job}_{index}_stdout.log'
            stderr_log_file_path = f'{FOLDER_HOME}/{index}/scripts/submit_{job}_{index}_stderr.log'

            with open(stdout_log_file_path, 'w') as stdout_log_file, open(stderr_log_file_path, 'w') as stderr_log_file:
                process = subprocess.Popen(f'bash {FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh &', 
                                           shell=True, stdout=stdout_log_file, stderr=stderr_log_file)

def extract_sequence_from_pdb(pdb_path):
    with open(pdb_path, "r") as pdb_file:
        for record in SeqIO.parse(pdb_file, "pdb-atom"):
            seq = str(record.seq)
    return seq

def find_highest_scoring_sequence(folder_path, parent_index, input_sequence_path, variables_json):
    """
    Identifies the highest scoring protein sequence from a set of generated PMPNN sequences,
    excluding the parent and WT sequence (except wildcard positions specified by DESIGN).

    Parameters:
    - folder_path (str): The path to the directory containing sequence files (/ProteinMPNN).
    - parent_index (str): The index of the parent protein sequence.
    - input_sequence_path (str): The path to a file containing the input sequence pattern,
      where 'X' represents wildcard positions that can match any character.
    - -------------------------------- GLOBAL variables used ----------------------------
    - WT (str): The wild type or reference protein identifier.
      

    Returns:
    - highest_scoring_sequence (str): The protein sequence with the highest score 
      that does not match the parent and WT.
    
    Note:
    This function parses .fa files to find sequences and their scores, and applies
    a regex pattern derived from the input sequence to filter sequences.
    It assumes the presence of 'global_score' within the sequence descriptor lines
    in the .fa file for scoring.
    """

    WT = variables_json['WT']
    # Construct the file path for the sequence data
    file_path = f'{folder_path}/seqs/{WT}_Rosetta_Relax_{parent_index}.fa'
    parent_seq_file = f'{folder_path}/Rosetta_Relax_{parent_index}.seq'
    
    # Read the parent sequence from its file
    with open(parent_seq_file, 'r') as file:
        parent_sequence = file.readline().strip()

    # Read the input sequence pattern and prepare it for regex matching
    with open(input_sequence_path, 'r') as file:
        input_sequence = file.readline().strip()
    pattern = re.sub('X', '.', input_sequence)  # Replace 'X' with regex wildcard '.'

    highest_score = 0
    highest_scoring_sequence = ''

    # Process the sequence file to find the highest scoring sequence
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                score_match = re.search('global_score=(\d+\.\d+)', line)
                if score_match:
                    score = float(score_match.group(1))
                    sequence = next(file, '').strip()  # Read the next line for the sequence
                    
                    # Check if the score is higher, the sequence is different from the parent,
                    # and does not match the input sequence pattern
                    if score > highest_score and sequence != parent_sequence and not re.match(pattern, sequence):
                        highest_score = score
                        highest_scoring_sequence = sequence

    # Return the highest scoring sequence found
    return highest_scoring_sequence

def save_cat_res_into_all_scores_df(all_scores_df, index, PDB_file_path, from_parent_struct=False):
    
    '''Finds the indices and names of the catalytic residue from <PDB_file_path> 
       Saves indices and residues into <all_scores_df> in row <index> as lists.
       To make sure these are saved and loaded as list, ";".join() and .split(";") should be used
       IF information is read from an input structure for design do not save cat_resn
       Returns the updated all_scores_df'''
    
    with open(PDB_file_path, 'r') as f: 
        PDB = f.readlines()
    
    remarks = [i for i in PDB if i[:10] == 'REMARK 666']

    cat_resis = []
    cat_resns = []

    for remark in remarks:
        cat_resis.append(str(int(remark[55:59])))

    for cat_resi in cat_resis:
        for line in PDB[len(remarks)+2:]:
            atomtype = line[12:16]
            if atomtype != " CA ": continue
            resi = str(int(line[22:26]))
            resn = line[17:20]
            if resi == cat_resi:
                cat_resns.append(resn)
                break

    all_scores_df.at[index, 'cat_resi'] = ";".join(cat_resis)
    # Only save the cat_resn if this comes from the designed structure, not from the input structure for design
    if not from_parent_struct:
        all_scores_df.at[index, 'cat_resn'] = ";".join(cat_resns)
    
    
    return all_scores_df

def generate_remark_from_all_scores_df(all_scores_df, index, variables_json):

    LIGAND = variables_json['LIGAND']

    remark = ''
    cat_resns = str(all_scores_df.at[index, 'cat_resn']).split(';')
    # Making sure resi is converted to int to avoid crash in Relax
    cat_resis = [int(float(x)) for x in str(all_scores_df.at[index, 'cat_resi']).split(';')]
    
    remarks = []

    for idx, (cat_resi, cat_resn) in enumerate(zip(cat_resis, cat_resns), start=1):
        remarks.append(f'REMARK 666 MATCH TEMPLATE X {LIGAND}    0 MATCH MOTIF A {cat_resn}{str(cat_resi).rjust(5)}  {idx}  1')
    return "\n".join(remarks)

def run_ProteinMPNN(parent_index, new_index, all_scores_df, variables_json):
    """
    Executes the ProteinMPNN pipeline for a given protein structure and generates
    new protein sequences with potentially higher functional scores. Also calls 
    ESMfold_RosettaRelax with the ProteinMPNN flag set to True.

    Parameters:
    - parent_index (str): The index of the parent protein variant.
    - new_index (str): The index assigned to the new protein variant.
    - all_scores_df (DataFrame): A DataFrame containing information for protein variants.
    - --------------------------------GLOBAL variables used (from variables.json) ----------------------------
    - FOLDER_HOME (str): The base directory where ProteinMPNN and related files are located.
    - WT (str): The wild type or reference protein identifier.
    - DESIGN (str): A string representing positions and types of amino acids to design with RosettaDesign.
    - ProteinMPNN_T (float): The sampling temperature for ProteinMPNN.
    - EXPLORE (bool): Flag to indicate whether exploration mode is enabled.
    - PMPNN_BIAS (float): The bias value for ProteinMPNN parent sequence retention.

    Returns:
    None: The function writes the highes scoring sequence to the specified directories.
    
    Note:
    This function assumes the ProteinMPNN toolkit is available and properly set up in the specified location.
    It involves multiple subprocess calls to Python scripts for processing protein structures and generating new sequences.
    """
    # Set GLOBALS
    FOLDER_HOME = variables_json['FOLDER_HOME']
    WT = variables_json['WT']
    DESIGN = variables_json['DESIGN']
    ProteinMPNN_T = variables_json['ProteinMPNN_T']
    EXPLORE = variables_json['EXPLORE']
    PMPNN_BIAS = variables_json['PMPNN_BIAS']

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
    highest_scoring_sequence = find_highest_scoring_sequence(protein_mpnn_folder, parent_index, input_sequence_path=f"{FOLDER_HOME}/input_sequence_with_X_as_wildecard.seq", variables_json=variables_json)

    # Save highest scoring sequence and prepare for ESMfold
    with open(os.path.join(protein_mpnn_folder, f"{WT}_{new_index}.seq"), "w") as f:
        f.write(highest_scoring_sequence)
    
    if highest_scoring_sequence:
        logging.info(f"Ran ProteinMPNN for index {parent_index} and found a new sequence with index {new_index}.")
    else:
        logging.error(f"Failed to find a new sequnce for index {parent_index} with ProteinMPNN.")
    
    all_scores_df = save_cat_res_into_all_scores_df(all_scores_df, new_index, pdb_file, from_parent_struct=False)
        
    # Run ESMfold Relax with the ProteinMPNN Flag
    return run_ESMfold_RosettaRelax(index=new_index, all_scores_df=all_scores_df, OnlyRelax=False, \
                             ProteinMPNN=True, ProteinMPNN_parent_index=parent_index, bash=True, variables_json=variables_json)

def run_ESMfold_RosettaRelax(index, all_scores_df, OnlyRelax=False, ProteinMPNN=False, PreMatchRelax=False,
                             ProteinMPNN_parent_index=0, cmd="", bash=False, EXPLORE=False, variables_json=None):
    
    # Set GLOBALS
    FOLDER_HOME = variables_json['FOLDER_HOME']
    FOLDER_INPUT = variables_json['FOLDER_INPUT']
    WT = variables_json['WT']
    DESIGN = variables_json['DESIGN']
    GRID = variables_json['GRID']
    BLUEPEBBLE = variables_json['BLUEPEBBLE']
    BACKGROUND_JOB = variables_json['BACKGROUND_JOB']
    ABBIE_LOCAL = variables_json['ABBIE_LOCAL']
    ROSETTA_PATH = variables_json['ROSETTA_PATH']
    LIGAND = variables_json['LIGAND']

    # Giving the ESMfold algorihm the needed inputs
    output_file = f'{FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_output_{index}.pdb'

    protein_mpnn_seq_file = f'{FOLDER_HOME}/{index}/ProteinMPNN/{WT}_{index}.seq'

    if os.path.exists(protein_mpnn_seq_file):
        sequence_file = protein_mpnn_seq_file
    else:
        logging.error(f"{protein_mpnn_seq_file} not present, but PMPNN was run prior to ESMfold.")
        return False


        
    # Make directories
    os.makedirs(f"{FOLDER_HOME}/{index}/ESMfold", exist_ok=True)
    os.makedirs(f"{FOLDER_HOME}/{index}/scripts", exist_ok=True)

        
    # Options for EXPLORE, accelerated script for testing
    ex = "-ex1 -ex2"
    if EXPLORE: ex = ""
        
    # Get Name of parent PDB
    PDBFile = f"{FOLDER_HOME}/{ProteinMPNN_parent_index}/{WT}_Rosetta_Relax_{ProteinMPNN_parent_index}.pdb"

    if not os.path.isfile(PDBFile):
        logging.error(f"{PDBFile} not present!")
        return False
                    
    # Get the pdb file from the last step and strip away ligand and hydrogens 
    cpptraj = f'''parm    {PDBFile}
trajin  {PDBFile}
strip   :{LIGAND}
strip   !@C,N,O,CA
trajout {FOLDER_HOME}/{index}/ESMfold/{WT}_CPPTraj_Apo_{index}.pdb
'''
    with open(f'{FOLDER_HOME}/{index}/ESMfold/CPPTraj_Apo_{index}.in','w') as f: f.write(cpptraj)

    # Get the pdb file from the last step and strip away everything except the ligand
    cpptraj = f'''parm    {PDBFile}
trajin  {PDBFile}
strip   !:{LIGAND}
trajout {FOLDER_HOME}/{index}/ESMfold/{WT}_CPPTraj_Lig_{index}.pdb
'''
    with open(f'{FOLDER_HOME}/{index}/ESMfold/CPPTraj_Lig_{index}.in','w') as f: f.write(cpptraj)

    # Get the ESMfold pdb file and strip away all hydrogens
    cpptraj = f'''parm    {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_output_{index}.pdb
trajin  {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_output_{index}.pdb
strip   !@C,N,O,CA
trajout {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_no_hydrogens_{index}.pdb
'''
    with open(f'{FOLDER_HOME}/{index}/ESMfold/CPPTraj_no_hydrogens_{index}.in','w') as f: f.write(cpptraj)

    # Align substrate and ESM prediction of scaffold without hydrogens
    cpptraj = f'''parm    {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_no_hydrogens_{index}.pdb
reference {FOLDER_HOME}/{index}/ESMfold/{WT}_CPPTraj_Apo_{index}.pdb [apo]
trajin    {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_no_hydrogens_{index}.pdb
rmsd      @CA ref [apo]
trajout   {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_aligned_{index}.pdb noter
'''
    with open(f'{FOLDER_HOME}/{index}/ESMfold/CPPTraj_aligned_{index}.in','w') as f: f.write(cpptraj) 
              
    if GRID:           extension = "linuxgccrelease"
    if BLUEPEBBLE:     extension = "serialization.linuxgccrelease"
    if BACKGROUND_JOB: extension = "serialization.linuxgccrelease"
    if ABBIE_LOCAL:    
        extension = "linuxgccrelease"
        bash_args = "OMP_NUM_THREADS=1"
    else:
        bash_args = ""
 
    cmd += f"""
    
{bash_args} python {FOLDER_HOME}/ESMfold.py {output_file} {sequence_file}

sed -i '/PARENT N\/A/d' {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_output_{index}.pdb
cpptraj -i {FOLDER_HOME}/{index}/ESMfold/CPPTraj_Apo_{index}.in           &> \
           {FOLDER_HOME}/{index}/ESMfold/CPPTraj_Apo_{index}.out
cpptraj -i {FOLDER_HOME}/{index}/ESMfold/CPPTraj_Lig_{index}.in           &> \
           {FOLDER_HOME}/{index}/ESMfold/CPPTraj_Lig_{index}.out
cpptraj -i {FOLDER_HOME}/{index}/ESMfold/CPPTraj_no_hydrogens_{index}.in  &> \
           {FOLDER_HOME}/{index}/ESMfold/CPPTraj_no_hydrogens_{index}.out
cpptraj -i {FOLDER_HOME}/{index}/ESMfold/CPPTraj_aligned_{index}.in       &> \
           {FOLDER_HOME}/{index}/ESMfold/CPPTraj_aligned_{index}.out

# Assemble the final protein
sed -i '/END/d' {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_aligned_{index}.pdb
# Return HETATM to ligand output and remove TER
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' {FOLDER_HOME}/{index}/ESMfold/{WT}_CPPTraj_Lig_{index}.pdb
"""
    
    input_extension_relax = ""
    if PreMatchRelax:
        extension_relax = "_APO"
        ## No ligand necessary so just use the aligned pdb from ESMfold
        cmd += f"""
cp {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_aligned_{index}.pdb \
   {FOLDER_HOME}/{index}/{WT}_ESMfold_{index}{extension_relax}.pdb
"""  

    else:
        extension_relax = ""
        remark = generate_remark_from_all_scores_df(all_scores_df, index, variables_json)
        with open(f'{FOLDER_HOME}/{index}/{WT}_ESMfold_{index}.pdb', 'w') as f: f.write(remark+"\n")
        cmd += f"""
cat {FOLDER_HOME}/{index}/ESMfold/{WT}_ESMfold_aligned_{index}.pdb >> {FOLDER_HOME}/{index}/{WT}_ESMfold_{index}.pdb
cat {FOLDER_HOME}/{index}/ESMfold/{WT}_CPPTraj_Lig_{index}.pdb     >> {FOLDER_HOME}/{index}/{WT}_ESMfold_{index}.pdb
sed -i '/TER/d' {FOLDER_HOME}/{index}/{WT}_ESMfold_{index}.pdb
"""
        
    cmd += f"""
# Run Rosetta Relax
{ROSETTA_PATH}/bin/rosetta_scripts.{extension} \
                -s                                        {FOLDER_HOME}/{index}/{WT}_ESMfold_{index}{extension_relax}.pdb \
                -extra_res_fa                             {FOLDER_INPUT}/{LIGAND}.params \
                -parser:protocol                          {FOLDER_HOME}/{index}/scripts/Rosetta_Relax_{index}.xml \
                -out:file:scorefile                       {FOLDER_HOME}/{index}/score_rosetta_relax.sc \
                -nstruct                                  1 \
                -ignore_zero_occupancy                    false \
                -corrections::beta_nov16                  true \
                -run:preserve_header                      true \
                -overwrite {ex}

# Rename the output file
mv {WT}_ESMfold_{index}{extension_relax}_0001.pdb {WT}_Rosetta_Relax_{index}{extension_relax}.pdb
sed -i '/        H  /d' {WT}_Rosetta_Relax_{index}{extension_relax}.pdb
"""
    
    if PreMatchRelax:
        extension_relax = "_APO"
        
        cmd += f"""
# Align relaxed ESM prediction of scaffold without hydrogens
cpptraj -i {FOLDER_HOME}/{index}/ESMfold/{WT}_Rosetta_Relax_aligned_{index}{extension_relax}.in           &> \
           {FOLDER_HOME}/{index}/ESMfold/{WT}_Rosetta_Relax_aligned_{index}{extension_relax}.out
sed -i '/END/d' {FOLDER_HOME}/{index}/{WT}_Rosetta_Relax_aligned_{index}{extension_relax}.pdb
"""  
        
        cpptraj = f'''parm    {FOLDER_HOME}/{index}/{WT}_Rosetta_Relax_{index}{extension_relax}.pdb [protein]
parm      {FOLDER_HOME}/{index}/ESMfold/{WT}_CPPTraj_Apo_{index}.pdb [reference]
reference {FOLDER_HOME}/{index}/ESMfold/{WT}_CPPTraj_Apo_{index}.pdb parm [reference] [apo]
trajin    {FOLDER_HOME}/{index}/{WT}_Rosetta_Relax_{index}{extension_relax}.pdb parm [protein]
rmsd      @CA ref [apo]
trajout   {FOLDER_HOME}/{index}/{WT}_Rosetta_Relax_aligned_{index}{extension_relax}.pdb noter
'''
        with open(f'{FOLDER_HOME}/{index}/ESMfold/{WT}_Rosetta_Relax_aligned_{index}{extension_relax}.in','w') as f: 
            f.write(cpptraj) 
    
        # Create the Rosetta_Relax.xml file
    repeats = "3"
    if EXPLORE: repeats = "1"
    Rosetta_Relax_xml = f"""
<ROSETTASCRIPTS>

    <SCOREFXNS>
    
        <ScoreFunction name      = "score"                   weights = "beta_nov16" >
            <Reweight scoretype  = "atom_pair_constraint"    weight  = "1" />
            <Reweight scoretype  = "angle_constraint"        weight  = "1" />
            <Reweight scoretype  = "dihedral_constraint"     weight  = "1" />
        </ScoreFunction> 
        
        <ScoreFunction name      = "score_final"             weights = "beta_nov16" >
            <Reweight scoretype  = "atom_pair_constraint"    weight  = "1" />
            <Reweight scoretype  = "angle_constraint"        weight  = "1" />
            <Reweight scoretype  = "dihedral_constraint"     weight  = "1" />
        </ScoreFunction>
        
    </SCOREFXNS>
       
    <MOVERS>
                                  
        <FastRelax  name="mv_relax" disable_design="false" repeats="{repeats}" /> 
"""
    if not PreMatchRelax: Rosetta_Relax_xml += f"""
        <AddOrRemoveMatchCsts     name="mv_add_cst" 
                                  cst_instruction="add_new" 
                                  cstfile="{FOLDER_INPUT}/{LIGAND}_{WT}_enzdes_planar.cst" />

"""
    Rosetta_Relax_xml += f"""

        <InterfaceScoreCalculator   name                   = "mv_inter" 
                                    chains                 = "X" 
                                    scorefxn               = "score_final" />
    </MOVERS>
    
    <PROTOCOLS>  

        <Add mover_name="mv_relax" />
"""
    if not PreMatchRelax: Rosetta_Relax_xml += f"""                                  
        <Add mover_name="mv_add_cst" />       
        <Add mover_name="mv_inter" />
"""
    Rosetta_Relax_xml += f"""
    </PROTOCOLS>
    
</ROSETTASCRIPTS>
"""
    # Write the Rosetta_Relax.xml to a file
    with open(f'{FOLDER_HOME}/{index}/scripts/Rosetta_Relax_{index}.xml', 'w') as f:
        f.writelines(Rosetta_Relax_xml)      
        
    if ProteinMPNN:
        with open(f'{FOLDER_HOME}/{index}/scripts/ProteinMPNN_ESMfold_Rosetta_Relax_{index}.sh', 'w') as file:
            file.write(cmd)
        logging.info(f"Run Ligand/ProteinMPNN for index {index} based on index {ProteinMPNN_parent_index}.")
        submit_job(index=index, job="ProteinMPNN_ESMfold_Rosetta_Relax", bash=bash, variables_json=variables_json)

    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ProteinMPNN and ESMfold Rosetta Relax workflows.")
    parser.add_argument("--index", type=int, required=True, help="Index for the current run.")
    parser.add_argument("--new_index", type=int, required=True, help="New index for the current run.")
    parser.add_argument("--home_folder", type=str, required=True, help="Path to the home folder.")
    args = parser.parse_args()

    # Read the variables.json file
    variables_json = f'{args.home_folder}/variables.json'
    with open(variables_json, 'r') as file:
        variables = json.load(file)
    
    #add home folder to variables
    variables['FOLDER_HOME'] = args.home_folder
    variables['FOLDER_INPUT'] = os.path.join(os.path.dirname(variables['FOLDER_HOME']), 'Input')

    all_scores_df = pd.read_csv(f'{args.home_folder}/all_scores.csv')

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(filename=f'{variables["FOLDER_HOME"]}.log', level=logging.DEBUG, format=log_format, datefmt=date_format)

    # Run the ProteinMPNN process
    success = run_ProteinMPNN(parent_index=args.index, new_index=args.new_index, all_scores_df=all_scores_df, variables_json=variables)

    if success:
        logging.info(f"Process completed successfully for index {args.index}.")
    else:
        logging.error(f"Process failed for index {args.index}.")

if __name__ == "__main__":
    main()
