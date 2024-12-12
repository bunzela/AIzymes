import os
import subprocess
import numpy as np
import pandas as pd
import pickle as pkl

from helper_001               import *

def prepare_efields(self, index:str, cmd:str):
    """
    Calculate electric fields for structure in {index}.
    
    Args:
    index (str): The index of the protein variant for whcih efields are to be calculated.
    cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
    cmd (str): Command to be exected by run_design using submit_job.
    """
   
    pdb = [line.strip("#").strip() for line in cmd.splitlines() if line.startswith("###")]
    pdb = [line for line in pdb if line in ["RosettaDesign", "RosettaRelax"]] # "ProteinMPNN", "LigandMPNN", 
    pdb = f'{self.WT}_{pdb[0]}_{index}'
        
    filename_in = f'{self.FOLDER_HOME}/{index}/{pdb}'
    filename_out = f'{self.FOLDER_HOME}/{index}/ElectricFields/{pdb}'
    
    # Make tleap files to generate input
    with open(f"{filename_out}_tleap.in", "w") as f:
        f.write(f"""source leaprc.protein.ff19SB 
    source leaprc.gaff
    loadamberprep   {self.FOLDER_INPUT}/5TS.prepi
    loadamberparams {self.FOLDER_INPUT}/5TS.frcmod
    mol = loadpdb {filename_out}.pdb
    saveamberparm mol {filename_out}.parm7 {filename_out}.rst7
    quit
    """)
            
    cmd += f"""### ElectricFields ###

# Delete everything after 'CONECT' and remove hydrogens 
sed -n '/CONECT/q;p' {filename_in}.pdb > \\
                     {filename_out}.pdb
sed -i '/ H /d'      {filename_out}.pdb

# Make AMBER files
tleap -f             {filename_out}_tleap.in > \\
                     {filename_out}_tleap.out
mv leap.log          {filename_out}_tleap.log

# Add field calculation command
python   {self.FIELD_TOOLS} \\
-nc      {filename_out}.rst7 \\
-parm    {filename_out}.parm7 \\
-out     {filename_out}_fields.pkl \\
-target  {self.FOLDER_PARENT}/field_target.dat \\
-solvent WAT


"""
    
    return cmd

def get_efields_score(self, index, score_type):

    with open(f"{self.FOLDER_HOME}/{index}/ElectricFields/{self.WT}_{score_type}_{index}_fields.pkl", "rb") as f:
        FIELDS = pkl.load(f)

    bond_field = FIELDS[f':5TS@C9_:5TS@H04']['Total'] # minus field of base
    all_fields = FIELDS[f':5TS@C9_:5TS@H04']

    return bond_field[0], all_fields

def update_efieldsdf(self, index:int, index_efields_dict:dict):
    '''Adds a new row to "{FOLDER_HOME}/electric_fields.csv" containing the electric fields 
    generated by FieldTools.py for all residues in the protein'''

    no_residues = len(index_efields_dict)-4

    gen_headers = ["Total","Protein","Solvent","WAT"]
    resi_headers = [f"RESI_{idx}" for idx in range(1,no_residues+1)]
    headers = gen_headers + resi_headers

    fields_list = [field[0] for field in index_efields_dict.values()]

    if not os.path.isfile(f"{self.FOLDER_HOME}/electric_fields.csv"):       
        fields_df = pd.DataFrame([fields_list], columns=headers, index=[index])
        fields_df.to_csv(f"{self.FOLDER_HOME}/electric_fields.csv") 

    else:
        fields_df = pd.read_csv(f"{self.FOLDER_HOME}/electric_fields.csv", index_col=0)
        new_row_df = pd.DataFrame([fields_list], columns=headers, index=[index])
        fields_df = pd.concat([fields_df, new_row_df])
        fields_df.sort_index(inplace=True)
        fields_df.to_csv(f"{self.FOLDER_HOME}/electric_fields.csv") 
