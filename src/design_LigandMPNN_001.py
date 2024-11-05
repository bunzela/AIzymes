
"""
NOTE: NOT WORKING YET!

Integrates LigandMPNN for generating protein sequences adapted to specific ligand contexts.

Functions:
    prepare_LigandMPNN: Executes LigandMPNN steps to adapt protein designs with ligand binding considerations.

Modules Required:
    helper_001
"""
def prepare_LigandMPNN(self, 
                       parent_index, 
                       new_index,
                       input_suffix,
                       cmd,
                       parent_done):
    """
    Executes the LigandMPNN pipeline for a given protein-ligand structure and generates
    new protein sequences with potentially higher functional scores considering the ligand context.

    Parameters:
    - parent_index (str): The index of the parent protein variant.
    - new_index (str): The index assigned to the new protein variant.
    - all_scores_df (DataFrame): A DataFrame containing information for protein variants.
    """

    PDB_input = get_PDB_in(self, parent_index, input_suffix, parent_done)
    
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