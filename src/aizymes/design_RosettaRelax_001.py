"""
Prepares commands to run RosettaRelax methods to refine protein structures.

Functions:
    prepare_RosettaRelax: Sets up commands for RosettaRelax job submission.
"""

import os 
              
def prepare_RosettaRelax(self, 
                         index,  
                         cmd,
                         PreMatchRelax=False # Placeholder for when RosettaMatch gets implemented! Should be removed!
                        ):
    """
    Relaxes protein structure in {index} using RosettaRelax.
    
    Parameters:
    index (str): The index of the protein variant to be relaxed.
    cmd (str): collection of commands to be run, this script wil append its commands to cmd
    
    Optional Parameters:
    PreMatchRelax (bool): True if ESMfold to be run without ligand (prior to RosettaMatch).

    """
    
    filename = f'{self.FOLDER_DESIGN}/{index}'
        
    # Make directories
    os.makedirs(f"{filename}/scripts", exist_ok=True)
    
    # Options for EXPLORE, accelerated script for testing
    if self.EXPLORE:
        repeats = "1"
        ex = ""
    else:
        repeats = "3"
        ex = "-ex1 -ex2"

    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"]
       
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
        RosettaRelax_xml += f"""
            <AddOrRemoveMatchCsts     name="mv_add_cst" 
                                  cst_instruction="add_new" 
                                  cstfile="{self.FOLDER_PARENT}/{self.CST_NAME}.cst" />

"""
        
    RosettaRelax_xml += f"""

        <InterfaceScoreCalculator   name                   = "mv_inter" 
                                    chains                 = "X" 
                                    scorefxn               = "score_final" />
    </MOVERS>
    
    <PROTOCOLS>  
"""

    if 'HEM' in self.LIGAND:
        RosettaRelax_xml += f"""                                  
        <Add mover_name="mv_add_cst" /> """
        
    RosettaRelax_xml += f"""
        <Add mover_name="mv_relax" />
"""

    if self.CST_NAME is not None and 'HEM' not in self.LIGAND:
        RosettaRelax_xml += f"""                                  
        <Add mover_name="mv_add_cst" /> """
    RosettaRelax_xml += f""" 
        <Add mover_name="mv_inter" />
"""
        
    RosettaRelax_xml += f"""
    </PROTOCOLS>
    
</ROSETTASCRIPTS>
"""
    
    cmd += f"""### RosettaRelax ###
"""
        
    # Write the RosettaRelax.xml to a file
    with open(f'{filename}/scripts/RosettaRelax_{index}.xml', 'w') as f:
        f.writelines(RosettaRelax_xml)   
  
    cmd += f"""
# Run Rosetta Relax
{self.ROSETTA_PATH}/bin/rosetta_scripts.{self.rosetta_ext} \\
    -s                                        {PDB_input}.pdb \\
    -parser:protocol                          {filename}/scripts/RosettaRelax_{index}.xml \\
    -out:file:scorefile                       {filename}/score_RosettaRelax.sc \\
    -nstruct                                  1 \\
    -ignore_zero_occupancy                    false \\
    -corrections::beta_nov16                  true \\
    -run:preserve_header                      true """
    if self.LIGAND not in ['HEM']:
        cmd += f"""\\
    -extra_res_fa                             {self.FOLDER_INPUT}/{self.LIGAND}.params """
    cmd += f"""\\
    -overwrite {ex}

# Cleanup
mv {self.FOLDER_DESIGN}/{index}/{os.path.basename(PDB_input)}_0001.pdb \\
   {self.FOLDER_DESIGN}/{index}/{self.WT}_RosettaRelax_{index}.pdb
sed -i '/        H  /d' {self.FOLDER_DESIGN}/{index}/{self.WT}_RosettaRelax_{index}.pdb
"""
    
    return cmd