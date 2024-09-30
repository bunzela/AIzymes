import logging

from helper_001               import *

from design_match_001         import *
from design_ProteinMPNN_001   import *
from design_LigandMPNN_001    import *
from design_RosettaDesign_001 import *
from design_ESMfold_001       import *
from design_RosettaRelax_001  import *

def run_design(self, 
               parent_index, 
               new_index,
               design_steps,
               parent_done=True,
               PreMatchRelax=False,
               bash=False
              ):
    
    cmd = ""
        
    for design_step in design_steps:
        
        if design_step == "RosettaDesign":
            cmd = prepare_RosettaDesign(self, 
                                        parent_index,
                                        new_index,
                                        "RosettaRelax", 
                                        cmd,
                                        parent_done)
            logging.debug(f"Run RosettaDesign for index {new_index} based on index {parent_index}.")
            
        if design_step == "RosettaRelax":
            
            cmd = prepare_RosettaRelax(self, 
                                       parent_index, 
                                       "ESMfold",
                                       "RosettaDesign",
                                       cmd,
                                       PreMatchRelax)
            logging.debug(f"Run RosettaRelax for index {parent_index}.")
            
        if design_step == "ESMfold":
            
            cmd = prepare_ESMfold(self, 
                                  parent_index, 
                                  "RosettaDesign", 
                                  cmd,
                                  PreMatchRelax)
            logging.debug(f"Run ESMfold for index {parent_index}.")
     
    # Write the shell command to a file and submit job                
    job = "_".join(design_steps)
    with open(f'{self.FOLDER_HOME}/{new_index}/scripts/{job}_{new_index}.sh','w') as file: file.write(cmd)
    submit_job(self, index=new_index, job=job, ram=2, bash=bash)
    
        
# --- OLD STUFF HERAFTER move somewhere else!
    
def find_highest_scoring_sequence(folder_path, parent_index, input_sequence_path):
    """
    Identifies the highest scoring protein sequence from a set of generated PNPNN sequences,
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

def find_highest_confidence_sequence(fa_file_path, output_seq_file_path):
    """
    Parses a .fa file to find the sequence with the highest overall confidence and writes it to a .seq file.

    Parameters:
    - fa_file_path (str): Path to the .fa file generated by LigandMPNN.
    - output_seq_file_path (str): Path where the .seq file should be saved.
    """
    highest_confidence = 0.0
    highest_confidence_sequence = None
    current_confidence = 0.0

    with open(fa_file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                # Extract overall confidence from the header line
                match = re.search('overall_confidence=([0-9.]+)', line)
                if match:
                    current_confidence = float(match.group(1))
            else:
                # Sequence line
                current_sequence = line.strip()
                if current_confidence > highest_confidence:
                    highest_confidence = current_confidence
                    highest_confidence_sequence = current_sequence

    # Write the highest confidence sequence to a .seq file
    if highest_confidence_sequence:
        with open(output_seq_file_path, 'w') as output_file:
            output_file.write(highest_confidence_sequence)
        logging.info(f"Extracted sequence with highest confidence: {highest_confidence} to {output_seq_file_path}")
    else:
        logging.error(f"No sequence found with overall confidence for {fa_file_path}.")

def run_LigandMPNN(parent_index, new_index, all_scores_df):
    """
    Executes the LigandMPNN pipeline for a given protein-ligand structure and generates
    new protein sequences with potentially higher functional scores considering the ligand context.

    Parameters:
    - parent_index (str): The index of the parent protein variant.
    - new_index (str): The index assigned to the new protein variant.
    - all_scores_df (DataFrame): A DataFrame containing information for protein variants.
    """

    # Testing LMPNN_submit.py
    cmd = f'python {os.getcwd()}/LMPNN_submit.py --index {parent_index} --new_index {new_index} --home_folder {FOLDER_HOME}'
    with open(f'{FOLDER_HOME}/{new_index}/scripts/LMPNN_ESM_Relax_{new_index}.sh','w') as file: file.write(cmd)
    submit_job(index=new_index, job="LMPNN_ESM_Relax")
    print(f"Run LMPNN ESM Relax on index {new_index}, based on parent index {parent_index}")
    return

    # Ensure LigandMPNN is available
    if not os.path.exists(f'{FOLDER_HOME}/../LigandMPNN'):
        logging.error(f"LigandMPNN not installed in {FOLDER_HOME}/LigandMPNN.")
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
    cat_resi = int(all_scores_df.at[parent_index, 'cat_resi'])
    fixed_residues = f"A{cat_resi}"

    # Run LigandMPNN
    run_command([
        "python", os.path.join(ligand_mpnn_path, "run.py"),
        "--model_type", "ligand_mpnn",
        "--temperature", LMPNN_T,
        "--seed", "37",
        "--pdb_path", os.path.join(ligand_mpnn_folder, f"{WT}_Rosetta_Relax_{parent_index}.pdb"),
        "--out_folder", ligand_mpnn_folder,
        #"--pack_side_chains", "1",
        "--number_of_packs_per_design", "4",
        "--fixed_residues", fixed_residues
    ], cwd=ligand_mpnn_path)

    find_highest_confidence_sequence(f"{FOLDER_HOME}/{new_index}/LigandMPNN/seqs/{WT}_Rosetta_Relax_{parent_index}.fa",
                                    f"{FOLDER_HOME}/{new_index}/LigandMPNN/{WT}_{new_index}.seq")

    # Update all_scores_df

    logging.info(f"Ran LigandMPNN for index {parent_index} and generated index {new_index}.")

    all_scores_df = save_cat_res_into_all_scores_df(all_scores_df, new_index, pdb_file, from_parent_struct=False)

    run_ESMfold_RosettaRelax(index=new_index, all_scores_df=all_scores_df, OnlyRelax=False, \
                             ProteinMPNN=True, ProteinMPNN_parent_index=parent_index, EXPLORE=False)

    # Save updates to all_scores_df
    #save_all_scores_df(all_scores_df)