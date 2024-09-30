import logging
import os
import shutil 
import subprocess  

from helper_001               import *

def prepare_ESMfold(self, 
                    index, 
                    input_suffix,
                    cmd, 
                    PreMatchRelax):
    """
    Predicts structure of sequence in {index} using ESMfold.
    
    Parameters:
    - index (str): The index of the protein variant to be predicted.
    
    Optional parameters:
    - PreMatchRelax (bool): True if ESMfold to be run without ligand (prior to RosettaMatch).

    Returns:
    - cmd (str): Command to be exected by run_design using submit_job.
    """
    
    filename = f'{self.FOLDER_HOME}/{index}'
    
    # Giving the ESMfold algorihm the needed inputs
    output_file = f'{filename}/{self.WT}_ESMfold_{index}.pdb'
    sequence_file = f'{filename}/ESMfold/{self.WT}_{index}.seq'
        
    # Make directories
    os.makedirs(f"{filename}/ESMfold", exist_ok=True)
    os.makedirs(f"{filename}/scripts", exist_ok=True)
        
    # Select parent PDB
    PDBfile = f"{filename}/{self.WT}_{input_suffix}_{index}.pdb"
    
    # Make sequence file
    if not os.path.isfile(PDBfile):
        logging.error(f"{PDBfile} not present!")
        print(f"Error {PDBfile} not present!")
        return False
    seq = extract_sequence_from_pdb(PDBfile)
    with open(sequence_file,"w") as f: f.write(seq)
       
    cmd += f"""
    
{self.bash_args} python {self.FOLDER_HOME}/ESMfold.py {output_file} {sequence_file}

sed -i '/PARENT N\/A/d' {output_file}
"""        
    return cmd