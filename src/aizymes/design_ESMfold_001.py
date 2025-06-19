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

from helper_002               import *

def prepare_ESMfold(self, 
                    index, 
                    cmd,
                    gpu_id = None):
    """
    Predicts structure of sequence in {index} using ESMfold. Uses ligand coordinates from previous RosettaDesign.
    
    Parameters:
    index (str): The index of the protein variant to be predicted.
    cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
    cmd (str): Command to be exected by run_design using submit_job.
    """
    filename = f'{self.FOLDER_DESIGN}/{index}'
        
    # Make directories
    os.makedirs(f"{filename}/scripts", exist_ok=True)
    os.makedirs(f"{filename}/ESMFold", exist_ok=True)

    working_dir_path = f"{filename}/ESMFold/{self.WT}_{index}"

    # Giving the ESMfold algorihm the needed inputs
    output_file = f'{working_dir_path}_ESMfold_bb.pdb'
    sequence_file = f'{self.FOLDER_DESIGN}/{index}/{self.WT}_{index}.seq'
        
    cmd += f"""### ESMfold ###
"""

    if  gpu_id != None:
        cmd += f"""
export CUDA_VISIBLE_DEVICES={gpu_id}
"""
        
    cmd += f"""
{self.bash_args}python {self.FOLDER_PARENT}/ESMfold.py \
--sequence_file {sequence_file} \
--output_file   {output_file} 

sed -i '/PARENT N\/A/d' {output_file}"""
    
    # Add the ligand back in after running ESMfold for backbone
    PDBfile_ligand = self.all_scores_df.at[int(index), "step_input_variant"]

    # Get the pdb file from the last step and strip away ligand and hydrogens 
    cpptraj = f'''parm {PDBfile_ligand}.pdb
trajin  {PDBfile_ligand}.pdb
strip   :{self.LIGAND}
strip   !@C,N,O,CA
trajout {working_dir_path}_old_bb.pdb
'''
    with open(f'{working_dir_path}_CPPTraj_old_bb.in','w') as f: f.write(cpptraj)

    # Get the pdb file from the last step and strip away everything except the ligand
    cpptraj = f'''parm    {PDBfile_ligand}.pdb
trajin  {PDBfile_ligand}.pdb
strip   !:{self.LIGAND}
trajout {working_dir_path}_lig.pdb
'''
    with open(f'{working_dir_path}_CPPTraj_lig.in','w') as f: f.write(cpptraj)

    # Get the ESMfold pdb file and strip away all hydrogens
    cpptraj = f'''parm {working_dir_path}_ESMfold_bb.pdb
trajin  {working_dir_path}_ESMfold_bb.pdb
strip   !@C,N,O,CA
trajout {working_dir_path}_new_bb.pdb
'''
    with open(f'{working_dir_path}_CPPTraj_new_bb.in','w') as f: f.write(cpptraj)

    # Align substrate and ESM prediction of scaffold without hydrogens
    cpptraj = f'''parm    {working_dir_path}_new_bb.pdb
reference {working_dir_path}_old_bb.pdb [ref]
trajin    {working_dir_path}_new_bb.pdb
rmsd      @CA ref [ref]
trajout   {working_dir_path}_aligned.pdb noter
'''
    with open(f'{working_dir_path}_CPPTraj_aligned.in','w') as f: f.write(cpptraj) 
 
    cmd += f"""
    
cpptraj -i {working_dir_path}_CPPTraj_old_bb.in &> \
           {working_dir_path}_CPPTraj_old_bb.out
cpptraj -i {working_dir_path}_CPPTraj_lig.in &> \
           {working_dir_path}_CPPTraj_lig.out
cpptraj -i {working_dir_path}_CPPTraj_new_bb.in &> \
           {working_dir_path}_CPPTraj_new_bb.out
cpptraj -i {working_dir_path}_CPPTraj_aligned.in &> \
           {working_dir_path}_CPPTraj_aligned.out

# Cleanup structures
sed -i '/END/d' {working_dir_path}_aligned.pdb
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' {working_dir_path}_lig.pdb
"""
        
    remark = generate_remark_from_all_scores_df(self, index)
    with open(f'{working_dir_path}_input.pdb', 'w') as f: f.write(remark+"\n")
    cmd += f"""# Assemble structure
cat {working_dir_path}_aligned.pdb >> {working_dir_path}_input.pdb
cat {working_dir_path}_lig.pdb     >> {working_dir_path}_input.pdb
sed -i '/TER/d' {working_dir_path}_input.pdb

cat {working_dir_path}_input.pdb > {self.FOLDER_DESIGN}/{index}/{self.WT}_ESMfold_{index}.pdb
""" 
        
    if self.REMOVE_TMP_FILES:
        cmd += f"""
# Removing temporary directory
rm -r ESMFold
"""
        

    return cmd