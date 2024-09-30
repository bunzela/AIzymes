import logging
import os
import shutil 
import subprocess  

from helper_001               import *
              
def prepare_RosettaRelax(self, 
                         index,  
                         input_suffix,
                         input_ligand_suffix,
                         cmd, 
                         PreMatchRelax=False):
    """
    Relaxes protein structure in {index} using RosettaRelax.
    
    Parameters:
    - index (str): The index of the protein variant to be relaxed.
    - input_suffix (str): 
    - input_ligand_suffix (str): 
    - cmd (str): collection of commands to be run, this script wil append its commands to cmd
    
    Optional parameters:
    - PreMatchRelax (bool): True if ESMfold to be run without ligand (prior to RosettaMatch).

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
      
    # Select parent PDB
    if PreMatchRelax: 
        PDBfile            = f"{self.FOLDER_INPUT}/{self.WT}"
        PDBfile_out        = f"{filename}/RosettaRelax/{self.WT}_{input_suffix}_{index}"
    else:
        PDBfile            = f"{filename}/{self.WT}_{input_suffix}_{index}"
        PDBfile_out        = f"{filename}/RosettaRelax/{self.WT}_{input_suffix}_{index}"
        PDBfile_ligand     = f"{filename}/{self.WT}_{input_ligand_suffix}_{index}"
        PDBfile_ligand_out = f"{filename}/RosettaRelax/{self.WT}_{input_ligand_suffix}_{index}"
    
    # Get the pdb file from the last step and strip away ligand and hydrogens 
    cpptraj = f'''parm    {PDBfile_ligand}.pdb
trajin  {PDBfile_ligand}.pdb
strip   :{self.LIGAND}
strip   !@C,N,O,CA
trajout {PDBfile_ligand_out}_bb.pdb
'''
    with open(f'{PDBfile_ligand_out}_CPPTraj_bb.in','w') as f: f.write(cpptraj)

    # Get the pdb file from the last step and strip away everything except the ligand
    cpptraj = f'''parm    {PDBfile_ligand}.pdb
trajin  {PDBfile_ligand}.pdb
strip   !:{self.LIGAND}
trajout {PDBfile_ligand_out}_lig.pdb
'''
    with open(f'{PDBfile_ligand_out}_CPPTraj_lig.in','w') as f: f.write(cpptraj)

    # Get the ESMfold pdb file and strip away all hydrogens
    cpptraj = f'''parm    {PDBfile}.pdb
trajin  {PDBfile}.pdb
strip   !@C,N,O,CA
trajout {PDBfile_out}_bb.pdb
'''
    with open(f'{PDBfile_out}_CPPTraj_bb.in','w') as f: f.write(cpptraj)

    # Align substrate and ESM prediction of scaffold without hydrogens
    cpptraj = f'''parm    {PDBfile_out}_bb.pdb
reference {PDBfile_ligand_out}_bb.pdb [ref]
trajin    {PDBfile_out}_bb.pdb
rmsd      @CA ref [ref]
trajout   {PDBfile_out}_aligned.pdb noter
'''
    with open(f'{PDBfile_out}_CPPTraj_aligned.in','w') as f: f.write(cpptraj) 
 
    cmd += f"""
    
cpptraj -i {PDBfile_ligand_out}_CPPTraj_bb.in &> \
           {PDBfile_ligand_out}_CPPTraj_bb.out
cpptraj -i {PDBfile_ligand_out}_CPPTraj_lig.in &> \
           {PDBfile_ligand_out}_CPPTraj_lig.out
cpptraj -i {PDBfile_out}_CPPTraj_bb.in &> \
           {PDBfile_out}_CPPTraj_bb.out
cpptraj -i {PDBfile_out}_CPPTraj_aligned.in &> \
           {PDBfile_out}_CPPTraj_aligned.out

# Cleanup aligned structure
sed -i '/END/d' {PDBfile_out}_aligned.pdb
# Cleanup ligand structure
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' {PDBfile_ligand_out}_lig.pdb
"""

    if PreMatchRelax:

        ## No ligand present so just use the aligned pdb from ESMfold
        cmd += f"""# Assemble structure
cp {PDBfile}_aligned.pdb {PDBfile_out}_input.pdb
"""  

    else:
        
        remark = generate_remark_from_all_scores_df(self, index)
        with open(f'{PDBfile_out}_input.pdb', 'w') as f: f.write(remark+"\n")
        cmd += f"""# Assemble structure
cat {PDBfile_out}_aligned.pdb        >> {PDBfile_out}_input.pdb
cat {PDBfile_ligand_out}_lig.pdb     >> {PDBfile_out}_input.pdb
sed -i '/TER/d' {PDBfile_out}_input.pdb
"""
        
    cmd += f"""
# Run Rosetta Relax
{self.ROSETTA_PATH}/bin/rosetta_scripts.{self.rosetta_ext} \
                -s                                        {PDBfile_out}_input.pdb \
                -extra_res_fa                             {self.FOLDER_INPUT}/{self.LIGAND}.params \
                -parser:protocol                          {filename}/scripts/RosettaRelax_{index}.xml \
                -out:file:scorefile                       {filename}/score_RosettaRelax.sc \
                -nstruct                                  1 \
                -ignore_zero_occupancy                    false \
                -corrections::beta_nov16                  true \
                -run:preserve_header                      true \
                -overwrite {ex}

# Rename the output file
mv {filename}/{self.WT}_{input_suffix}_{index}_input_0001.pdb {self.WT}_RosettaRelax_{index}.pdb
sed -i '/        H  /d' {self.WT}_RosettaRelax_{index}.pdb
"""

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
    if not PreMatchRelax: RosettaRelax_xml += f"""                                  
        <Add mover_name="mv_add_cst" />       
        <Add mover_name="mv_inter" />
"""
        
    RosettaRelax_xml += f"""
    </PROTOCOLS>
    
</ROSETTASCRIPTS>
"""
    # Write the RosettaRelax.xml to a file
    with open(f'{filename}/scripts/RosettaRelax_{index}.xml', 'w') as f:
        f.writelines(RosettaRelax_xml)      
       
    return cmd