"""
Design ESMfold Module

Manages the structure prediction of protein sequences using ESMfold within the AIzymes project.

Functions:
    - prepare_ESMfold: Prepares commands for ESMfold job submission.
"""
import logging
import os
import shutil 
import subprocess  

from helper_001               import *

def prepare_ESMfold(self, 
                    index, 
                    cmd):
    """
    Predicts structure of sequence in {index} using ESMfold.
    
    Parameters:
    index (str): The index of the protein variant to be predicted.
    cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
    cmd (str): Command to be exected by run_design using submit_job.
    """
        
    # Giving the ESMfold algorihm the needed inputs
    output_file = f'{self.FOLDER_HOME}/{index}/{self.WT}_ESMfold_{index}.pdb'
    sequence_file = f'{self.FOLDER_HOME}/{index}/{self.WT}_{index}.seq'
    
    # Make sequence file exist
    if not os.path.isfile(sequence_file):      
        logging.error(f"Sequence_file {sequence_file} not present!")
        return False
       
    cmd += f"""
    
{self.bash_args} python {self.FOLDER_PARENT}/ESMfold.py \
--sequence_file {sequence_file} \
--output_file   {output_file} 

sed -i '/PARENT N\/A/d' {output_file}
"""        
    return cmd