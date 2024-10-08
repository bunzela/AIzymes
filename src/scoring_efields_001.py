import os
import subprocess
import numpy as np
import pandas as pd
import pickle as pkl

def generate_AMBER_files(self, filename_in:str, filename_out:str):
    
    '''Uses tleap to create a .parm7 and .rst7 file from a pdb. Requires ambertools.
    Also requires 5TS.prepi and 5TS.frcmod in the INPUT folder
    TODO: Add script to generate these if not present.
    
    Parameters:
    - filename (str): The path to the pdb file to analyse without the file extension.'''

    #delete everything after 'CONECT' - otherwise Amber tries to read it as extra residues
    subprocess.run(['sed', '-n', '/CONECT/q;p', f"{filename_in}.pdb"],
                    stdout=open(f"{filename_out}_clean.pdb", "w"))
    
    #remove hydrogens 
    subprocess.run(['sed', '/ H /d', f"{filename_out}_clean.pdb"],
                    stdout=open(f"{filename_out}_noHclean.pdb", "w"))

    os.remove(f"{filename_out}_clean.pdb")

    with open("tleap.in", "w") as f:
            f.write(f"""source leaprc.protein.ff19SB 
    source leaprc.gaff
    loadamberprep   {self.FOLDER_INPUT}/5TS.prepi
    loadamberparams {self.FOLDER_INPUT}/5TS.frcmod
    mol = loadpdb {filename_out}_noHclean.pdb
    saveamberparm mol {filename_out}.parm7 {filename_out}.rst7
    quit
    """)
    subprocess.run(["tleap", "-s", "-f", "tleap.in"],
                   stdout=open(f"{filename_out}_tleap.out", "w"))

def calc_efields_score(self, pdb_path, cat_resi):
    
    '''Executes the FieldTools.py script to calculate the electric field across the C-H bond of 5TS.
    Requires a field_target.dat in the Input folder. Currently hard-coded based on 5TS
    TODO: Make this function agnostic to contents of field_target
    
    Parameters:
    - pdb_path (str): The path to the pdb structure to analyse - either from design or relax.
    
    Returns:
    - bond_field (float): The total electric field across the 5TS@C9_:5TS@H04 bond in MV/cm. (Currently hard coded to these atoms)
    - all_fields (dict): The components of the electric field across the 5TS@C9_:5TS@H04 bond per residue.'''

    def parse_pdb(pdb_file, cat_resi):
        with open(pdb_file, 'r') as f:
            lines = f.readlines()

        atoms = {}
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_seq = int(line[22:26].strip())
                
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                if res_name == '5TS' and atom_name == 'H4':
                    print(atom_name, res_name, chain_id, res_seq)
                    atoms['5TS@H4'] = np.array([x, y, z])
                elif res_seq == cat_resi:
                    print(atom_name, res_name, chain_id, res_seq)
                    if atom_name == 'OD1':
                        atoms['OD1'] = np.array([x, y, z])
                    elif atom_name == 'OD2':
                        atoms['OD2'] = np.array([x, y, z])

        return atoms
    
    def calculate_distances(atoms):
        h4 = atoms['5TS@H4']
        od1 = atoms['OD1']
        od2 = atoms['OD2']

        dist_od1 = np.linalg.norm(h4 - od1)
        dist_od2 = np.linalg.norm(h4 - od2)

        return dist_od1, dist_od2

    # atoms = parse_pdb(pdb_path, cat_resi)
    # dist_od1, dist_od2 = calculate_distances(atoms)

    # if dist_od1 < dist_od2:
    #     closer_oxygen = 'OD1'
    #     distal_oxygen = 'OD2'
    # else:
    #     closer_oxygen = 'OD2'
    #     distal_oxygen = 'OD1'

    # Define directory and filename
    filename_in, _ = os.path.splitext(pdb_path)     
    directory, filename_out = os.path.split(pdb_path)  
    os.makedirs(f'{directory}/efield', exist_ok=True) 
    filename_out, _ = os.path.splitext(filename_out)
    filename_out = f'{directory}/efield/{filename_out}'
    generate_AMBER_files(self, filename_in, filename_out)

    # Can be run with field_target_OD for C-O axis efield calculation
    subprocess.run(["python", f"{self.FIELD_TOOLS}", 
                    "-nc", f"{os.path.relpath(filename_out)}.rst7", 
                    "-parm", f"{os.path.relpath(filename_out)}.parm7", 
                    "-out", f"{os.path.relpath(filename_out)}_fields.pkl", 
                    "-target", "Input/field_target.dat",
                    "-solvent", "WAT"],
                    stdout=open(f"{os.path.relpath(filename_out)}_fields.out", "w"))
    

    # with open(f"{filename_out}_fields_OD.pkl", "rb") as f:
    #     FIELDS = pkl.load(f)
    
    with open(f"{filename_out}_fields.pkl", "rb") as f:
        FIELDS = pkl.load(f)

    bond_field = FIELDS[f':5TS@C9_:5TS@H04']['Total']
    all_fields = FIELDS[f':5TS@C9_:5TS@H04']

    # bond_field_co = FIELDS[f':5TS@C9_:{cat_resi}@{closer_oxygen}']['Total']
    # all_fields = FIELDS[f':5TS@C9_:{cat_resi}@{closer_oxygen}']

    #bond_field_do = FIELDS[f':5TS@C9_:{cat_resi}@{distal_oxygen}']['Total']
    #all_fields = FIELDS[f':5TS@C9_:{cat_resi}@{distal_oxygen}']

    #print(f"Distal: {cat_resi}@{distal_oxygen}", bond_field_do[0])

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
