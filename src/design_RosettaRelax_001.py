
"""
Executes RosettaRelax to refine protein structures and improve stability in the AIzymes project.

Functions:
    prepare_RosettaRelax: Sets up commands for RosettaRelax job submission.

Modules Required:
    helper_001
"""
import logging
import os
import shutil 
import subprocess  

from helper_002               import *
              
def prepare_RosettaRelax(self, 
                         index,  
                         cmd,
                         PreMatchRelax=False # Placeholder for when RosettaMatch gets implemented! Should be removed!
                        ):
    """
    Relaxes protein structure in {index} using RosettaRelax.
    
    Args:
    index (str): The index of the protein variant to be relaxed.
    cmd (str): collection of commands to be run, this script wil append its commands to cmd
    
    Optional Args:
    PreMatchRelax (bool): True if ESMfold to be run without ligand (prior to RosettaMatch).

    """
    
    filename = f'{self.FOLDER_HOME}/{index}'
        
    # Make directories
    os.makedirs(f"{filename}/scripts", exist_ok=True)
    os.makedirs(f"{filename}/RosettaRelax", exist_ok=True)
    
    # Options for EXPLORE, accelerated script for testing
    if self.EXPLORE:
        repeats = "1"
        ex = ""
    else:
        repeats = "3"
        ex = "-ex1 -ex2"
      
    #_, PDBfile, PDBfile_ligand = get_PDB_in(self, index)

    input_pdb_paths = get_PDB_in(self, index)
    PDBfile_in = input_pdb_paths['Relax_in']
        
    working_dir_path = f"{filename}/RosettaRelax/{self.WT}_{index}"
  
    # Create the RosettaRelax.xml file    
    RosettaRelax_xml = f"""
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
    
    # Add constraint if run is not used as PreMatchRelax
    if self.CST_NAME is not None:
        if not PreMatchRelax: RosettaRelax_xml += f"""
            <AddOrRemoveMatchCsts     name="mv_add_cst" 
                                  cst_instruction="add_new" 
                                  cstfile="{self.FOLDER_INPUT}/{self.CST_NAME}.cst" />

"""
        
    RosettaRelax_xml += f"""

        <InterfaceScoreCalculator   name                   = "mv_inter" 
                                    chains                 = "X" 
                                    scorefxn               = "score_final" />
    </MOVERS>
    
    <PROTOCOLS>  

        <Add mover_name="mv_relax" />
"""
    
    # Add constraint if run is not used as PreMatchRelax
    if self.CST_NAME is not None:
        if not PreMatchRelax: RosettaRelax_xml += f"""                                  
        <Add mover_name="mv_add_cst" />       
        <Add mover_name="mv_inter" />
"""
        
    RosettaRelax_xml += f"""
    </PROTOCOLS>
    
</ROSETTASCRIPTS>
"""
    
    cmd += f"""### RosettaRelax ###
    
# Assemble structure
cat {PDBfile_in}.pdb > {working_dir_path}_input.pdb
"""
        
    # Write the RosettaRelax.xml to a file
    with open(f'{filename}/scripts/RosettaRelax_{index}.xml', 'w') as f:
        f.writelines(RosettaRelax_xml)   
  
    cmd += f"""
# Run Rosetta Relax
{self.ROSETTA_PATH}/bin/rosetta_scripts.{self.rosetta_ext} \\
    -s                                        {working_dir_path}_input.pdb \\
    -parser:protocol                          {filename}/scripts/RosettaRelax_{index}.xml \\
    -out:file:scorefile                       {filename}/score_RosettaRelax.sc \\
    -nstruct                                  1 \\
    -ignore_zero_occupancy                    false \\
    -corrections::beta_nov16                  true \\
    -run:preserve_header                      true \\"""
    if self.LIGAND not in ['HEM']:
        cmd += f"""\\
    -extra_res_fa                             {self.FOLDER_INPUT}/{self.LIGAND}.params """
    cmd += f"""\\    
    -overwrite {ex}

# Rename the output file
mv {filename}/{self.WT}_{index}_input_0001.pdb {self.WT}_RosettaRelax_{index}.pdb
sed -i '/        H  /d' {self.WT}_RosettaRelax_{index}.pdb

"""
    
    return cmd