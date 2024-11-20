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

def find_highest_scoring_sequence(folder_path, parent_index, variables_json):
    """
    Identifies the highest scoring protein sequence from a set of generated LMPNN sequences,
    excluding the parent sequence.

    Parameters:
    - folder_path (str): The path to the directory containing sequence files (/LigandMPNN).
    - parent_index (str): The index of the parent protein sequence.
    - -------------------------------- GLOBAL variables used ----------------------------
    - WT (str): The wild type or reference protein identifier.
      
    Returns:
    - highest_scoring_sequence (str): The protein sequence with the highest score 
      that does not match the parent sequence.
    
    Note:
    This function parses .fa files to find sequences and their scores.
    It assumes the presence of 'global_score' within the sequence descriptor lines
    in the .fa file for scoring.
    """

    WT = variables_json['WT']
    # Construct the file path for the sequence data
    file_path = f'{folder_path}/seqs/{WT}_Rosetta_Relax_{parent_index}.fa'
    parent_seq_file = f'{folder_path}/{WT}_Rosetta_Relax_{parent_index}.seq'
    
    # Read the parent sequence from its file
    with open(parent_seq_file, 'r') as file:
        parent_sequence = file.readline().strip()

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
                    
                    # Check if the score is higher and the sequence is different from the parent
                    if score > highest_score and sequence != parent_sequence:
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

    # Ensure the column is of string type before assignment
    all_scores_df['cat_resi'] = all_scores_df['cat_resi'].astype(str)
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

def run_LigandMPNN(parent_index, new_index, all_scores_df, variables_json):
    """
    Executes the LigandMPNN pipeline for a given protein-ligand structure and generates
    new protein sequences with potentially higher functional scores considering the ligand context.

    Parameters:
    - parent_index (str): The index of the parent protein variant.
    - new_index (str): The index assigned to the new protein variant.
    - all_scores_df (DataFrame): A DataFrame containing information for protein variants.
    - variables_json (dict): A dictionary containing global variables.
    """
    # Set GLOBALS
    FOLDER_HOME = variables_json['FOLDER_HOME']
    WT = variables_json['WT']
    LMPNN_T = variables_json['LMPNN_T']
    EXPLORE = variables_json['EXPLORE']
    LMPNN_BIAS = variables_json['LMPNN_BIAS']

    # Ensure LigandMPNN is available
    if not os.path.exists(f'{FOLDER_HOME}/../LigandMPNN'):
        logging.error(f"LigandMPNN not installed in {FOLDER_HOME}/../LigandMPNN.")
        logging.error("Install using: git clone https://github.com/dauparas/LigandMPNN.git")
        return
    ligand_mpnn_path = f"{FOLDER_HOME}/../LigandMPNN"

    # Prepare file paths
    pdb_file = f"{FOLDER_HOME}/{parent_index}/{WT}_Rosetta_Relax_{parent_index}.pdb"
    if not os.path.isfile(pdb_file):
        logging.error(f"{pdb_file} not present!")
        return

    ligand_mpnn_folder = f"{FOLDER_HOME}/{new_index}/LigandMPNN"
    os.makedirs(ligand_mpnn_folder, exist_ok=True)
    shutil.copy(pdb_file, os.path.join(ligand_mpnn_folder, f"{WT}_Rosetta_Relax_{parent_index}.pdb"))

    # Extract catalytic residue information
    cat_resi = str(all_scores_df.at[parent_index, 'cat_resi']).split(';')
    fixed_residues = " ".join([f"A{resi}" for resi in cat_resi])

    # Construct bias_AA_per_residue.json
    parent_seq = extract_sequence_from_pdb(os.path.join(ligand_mpnn_folder, f"{WT}_Rosetta_Relax_{parent_index}.pdb"))
    parent_sequence_file = f"{ligand_mpnn_folder}/{WT}_Rosetta_Relax_{parent_index}.seq"
    with open(parent_sequence_file, 'w') as file:
        file.write(parent_seq)

    bias_dict = {f"A{i+1}": {aa: LMPNN_BIAS} for i, aa in enumerate(parent_seq)}
    bias_file_path = os.path.join(ligand_mpnn_folder, "bias_AA_per_residue.json")
    with open(bias_file_path, 'w') as bias_file:
        json.dump(bias_dict, bias_file)

    # Run LigandMPNN
    run_command([
        "python", os.path.join(ligand_mpnn_path, "run.py"),
        "--model_type", "ligand_mpnn",
        "--temperature", str(LMPNN_T),
        "--seed", "37",
        "--pdb_path", os.path.join(ligand_mpnn_folder, f"{WT}_Rosetta_Relax_{parent_index}.pdb"),
        "--out_folder", ligand_mpnn_folder,
        "--fixed_residues", fixed_residues,
        "--bias_AA_per_residue", bias_file_path,
        "--pack_side_chains", "1",
        "--number_of_packs_per_design", "1",
        "--pack_with_ligand_context", "1"
    ], cwd=ligand_mpnn_path)

    # Find highest confidence sequence
    highest_scoring_sequence = find_highest_scoring_sequence(
        ligand_mpnn_folder, parent_index, variables_json
    )

    # Save highest scoring sequence and prepare for ESMfold
    with open(os.path.join(ligand_mpnn_folder, f"{WT}_{new_index}.seq"), "w") as f:
        f.write(highest_scoring_sequence)

    if highest_scoring_sequence:
        logging.info(f"Ran LigandMPNN for index {parent_index} and found a new sequence with index {new_index}.")
    else:
        logging.error(f"Failed to find a new sequence for index {parent_index} with LigandMPNN.")

    all_scores_df = save_cat_res_into_all_scores_df(all_scores_df, new_index, pdb_file, from_parent_struct=False)

    # Run ESMfold Relax with the LigandMPNN Flag
    return run_ESMfold_RosettaRelax(
        index=new_index, all_scores_df=all_scores_df, OnlyRelax=False,
        LigandMPNN=True, MPNN_parent_index=parent_index, bash=True, variables_json=variables_json
    )

def run_ESMfold_RosettaRelax(index, all_scores_df, OnlyRelax=False, LigandMPNN=False, PreMatchRelax=False,
                             MPNN_parent_index=0, cmd="", bash=False, EXPLORE=False, variables_json=None):
    
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

    ligand_mpnn_seq_file = f'{FOLDER_HOME}/{index}/LigandMPNN/{WT}_Rosetta_Relax_{MPNN_parent_index}.seq'

    if os.path.exists(ligand_mpnn_seq_file):
        sequence_file = ligand_mpnn_seq_file
    else:
        logging.error(f"{ligand_mpnn_seq_file} not present, but LMPNN was run prior to ESMfold.")
        return False


        
    # Make directories
    os.makedirs(f"{FOLDER_HOME}/{index}/ESMfold", exist_ok=True)
    os.makedirs(f"{FOLDER_HOME}/{index}/scripts", exist_ok=True)

        
    # Options for EXPLORE, accelerated script for testing
    ex = "-ex1 -ex2"
    if EXPLORE: ex = ""
        
    # Get Name of parent PDB
    PDBFile = f"{FOLDER_HOME}/{MPNN_parent_index}/{WT}_Rosetta_Relax_{MPNN_parent_index}.pdb"

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
        
    if LigandMPNN:
        with open(f'{FOLDER_HOME}/{index}/scripts/LigandMPNN_ESMfold_Rosetta_Relax_{index}.sh', 'w') as file:
            file.write(cmd)
        logging.info(f"Run Ligand/ProteinMPNN for index {index} based on index {MPNN_parent_index}.")
        submit_job(index=index, job="LigandMPNN_ESMfold_Rosetta_Relax", bash=bash, variables_json=variables_json)

    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LigandMPNN and ESMfold Rosetta Relax workflows.")
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

    # Run the LigandMPNN process
    success = run_LigandMPNN(parent_index=args.index, new_index=args.new_index, all_scores_df=all_scores_df, variables_json=variables)

    if success:
        logging.info(f"Process completed successfully for index {args.index}.")
    else:
        logging.error(f"Process failed for index {args.index}.")

if __name__ == "__main__":
    main()
