
"""
Design Match Module

Provides functionalities for matching protein designs with specific constraints
and requirements in the AIzymes workflow.

Functions:
    - prepare_LigandMPNN: Executes the LigandMPNN pipeline for protein-ligand structure adaptation.

Modules Required:
    - helper_001
"""
def run_RosettaMatch(run_apo = False, EXPLORE=False, submit=False, bash=True):
        
    prepare_input_files()
    os.makedirs(FOLDER_MATCH, exist_ok=True)

    if run_apo:
        os.makedirs(f'{FOLDER_HOME}/{FOLDER_MATCH}/scripts', exist_ok=True)
        shutil.copyfile(f'{FOLDER_INPUT}/{WT}.pdb', f'{FOLDER_HOME}/{FOLDER_MATCH}/{WT}_{FOLDER_MATCH}_APO.pdb')
        run_Matcher_apo()
        print("RosettaMatch APO is done")
    else:
        if not os.path.isdir(f'{FOLDER_HOME}/{FOLDER_MATCH}/scripts'):
            run_ESMfold_RosettaRelax(FOLDER_MATCH, all_scores_df=None, PreMatchRelax=True, EXPLORE=EXPLORE) 
        elif not os.path.isfile(f'{FOLDER_HOME}/{FOLDER_MATCH}/{WT}_Rosetta_Relax_{FOLDER_MATCH}_APO.pdb'):
            print(f"ESMfold and Relax of {FOLDER_MATCH} still running.")
        elif not os.path.isdir(f'{FOLDER_HOME}/{FOLDER_MATCH}/matches'):
            run_Matcher()
        else:
            print("Matching is done")
            
def run_Matcher():
        
    cmd = f"""       
  
cd {FOLDER_HOME}/{FOLDER_MATCH}

echo C9 > {LIGAND}.central
echo {" ".join(MATCH.split(","))} > {LIGAND}.pos

{ROSETTA_PATH}/bin/gen_lig_grids.linuxgccrelease \
    -s                      {WT}_Rosetta_Relax_aligned_{FOLDER_MATCH}_APO.pdb ESMfold/{WT}_CPPTraj_Lig_{FOLDER_MATCH}.pdb \
    -extra_res_fa           {FOLDER_INPUT}/{LIGAND}.params \
    -grid_delta             0.5 \
    -grid_lig_cutoff        5.0 \
    -grid_bb_cutoff         2.25 \
    -grid_active_res_cutoff 15.0 \
    -overwrite 

mv {WT}_Rosetta_Relax_aligned_{FOLDER_MATCH}_APO.pdb_0.gridlig {WT}.gridlig
rm {WT}_Rosetta_Relax_aligned_{FOLDER_MATCH}_APO.pdb_0.pos 2>1

rm -r matches
mkdir matches
cd matches

{ROSETTA_PATH}/bin/match.linuxgccrelease \
    -s                                        ../{WT}_Rosetta_Relax_aligned_{FOLDER_MATCH}_APO.pdb \
    -match:lig_name                           {LIGAND} \
    -extra_res_fa                             {FOLDER_INPUT}/{LIGAND}.params \
    -match:geometric_constraint_file          {FOLDER_INPUT}/{LIGAND}_{WT}_enzdes_planar.cst \
    -match::scaffold_active_site_residues     ../{LIGAND}.pos \
    -match:required_active_site_atom_names    ../{LIGAND}.central \
    -match:active_site_definition_by_gridlig  ../{WT}.gridlig  \
    -match:grid_boundary                      ../{WT}.gridlig  \
    -gridligpath                              ../{WT}.gridlig  \
    -overwrite  \
    -output_format PDB  \
    -output_matches_per_group 1  \
    -consolidate_matches true 
""" 
    with open(f'{FOLDER_HOME}/{FOLDER_MATCH}/scripts/RosettaMatch_{FOLDER_MATCH}.sh', 'w') as file: file.write(cmd)
    logging.info(f"Run Rosetta_Match for index {FOLDER_MATCH}.")
    submit_job(FOLDER_MATCH, job="RosettaMatch", bash=False)

def run_Matcher_apo():
    # Combine the commands into a single script
    # The flags for gen_apo_grids imerically tuned to yield only relevant pos AA for the catalytic site (based on a NTF2 Fold)
    cmd = f"""
cd {FOLDER_HOME}/{FOLDER_MATCH}

echo C9 > {LIGAND}.central

# Generate grid and pos files using gen_apo_grids
{ROSETTA_PATH}/bin/gen_apo_grids.linuxgccrelease \
    -s {FOLDER_HOME}/{FOLDER_MATCH}/{WT}_{FOLDER_MATCH}_APO.pdb \
    -mute all \
    -unmute apps.pilot.wendao.gen_apo_grids \
    -chname off \
    -constant_seed \
    -ignore_unrecognized_res \
    -packstat:surface_accessibility \
    -packstat:cavity_burial_probe_radius 2.3 \
    -packstat:cluster_min_volume 30 \
    -packstat:min_cluster_overlap 1.0 \
    -packstat:min_cav_ball_radius 1.5 \
    -packstat:min_surface_accessibility 1.4 

# Use the largest cavity's grid and pos files to run the matcher
GRID_FILE=$(ls -1 {WT}_{FOLDER_MATCH}_APO.pdb_*.gridlig | head -n 1)
POS_FILE=$(ls -1 {WT}_{FOLDER_MATCH}_APO.pdb_*.pos | head -n 1)

rm -r matches
mkdir matches
cd matches

{ROSETTA_PATH}/bin/match.linuxgccrelease \
    -s                                        ../{WT}_{FOLDER_MATCH}_APO.pdb \
    -match:lig_name                           {LIGAND} \
    -extra_res_fa                             {FOLDER_INPUT}/{LIGAND}.params \
    -match:geometric_constraint_file          {FOLDER_INPUT}/{LIGAND}_enzdes.cst \
    -match::scaffold_active_site_residues     ../$POS_FILE \
    -match:required_active_site_atom_names    ../{LIGAND}.central \
    -match:active_site_definition_by_gridlig  ../$GRID_FILE \
    -match:grid_boundary                      ../$GRID_FILE \
    -gridligpath                              ../$GRID_FILE \
    -overwrite  \
    -output_format PDB  \
    -output_matches_per_group 1  \
    -consolidate_matches true 
"""
    # Write the combined script to a file
    with open(f'{FOLDER_HOME}/{FOLDER_MATCH}/scripts/RosettaMatch_{FOLDER_MATCH}.sh', 'w') as file:
        file.write(cmd)
    
    logging.info(f"Run combined gen_apo_grids and Rosetta_Match for index {FOLDER_MATCH}.")
    submit_job(FOLDER_MATCH, job="RosettaMatch", bash=False)


def separate_matches():
    parent_folder = f'{FOLDER_HOME}/{FOLDER_PARENT}'
    pdb_files = [f for f in os.listdir(parent_folder) if f.endswith('.pdb')]

    for pdb_file in pdb_files:
        pdb_path = os.path.join(parent_folder, pdb_file)
        with open(pdb_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('REMARK 666 MATCH TEMPLATE'):
                    parts = line.split()
                    catalytic_residue_index = parts[11]  # Extract the catalytic residue index
                    break

        # Create the destination folder if it doesn't exist
        dest_folder = os.path.join(parent_folder, catalytic_residue_index)
        os.makedirs(dest_folder, exist_ok=True)

        # Copy the PDB file to the destination folder instead of moving it
        shutil.copy(pdb_path, os.path.join(dest_folder, pdb_file))
    
