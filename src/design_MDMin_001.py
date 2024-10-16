import logging
import os
import shutil 
import subprocess  

from helper_001 import get_PDB_in, generate_remark_from_all_scores_df
              
def prepare_MDMin(self, 
                  index,  
                  cmd,
                  ):
    """
    Minimises protein structure in {index} using Amber.
    
    Parameters:
    - index (str): The index of the protein variant to be relaxed.
    - cmd (str): collection of commands to be run, this script wil append its commands to cmd
    """
    
    filename = f'{self.FOLDER_HOME}/{index}'
        
    # Make directories
    os.makedirs(f"{filename}/scripts", exist_ok=True)
    os.makedirs(f"{filename}/MDMin", exist_ok=True)
      
    _, PDBfile, PDBfile_ligand = get_PDB_in(self, index)
        
    PDBfile_out = f"{filename}/MDMin/{self.WT}_{index}"
    
    # Get the pdb file from the last step and strip away ligand and hydrogens 
    cpptraj = f'''parm    {PDBfile_ligand}.pdb
trajin  {PDBfile_ligand}.pdb
strip   :{self.LIGAND}
strip   !@C,N,O,CA
trajout {PDBfile_out}_old_bb.pdb
'''
    with open(f'{PDBfile_out}_CPPTraj_old_bb.in','w') as f: f.write(cpptraj)

    # Get the pdb file from the last step and strip away everything except the ligand
    cpptraj = f'''parm    {PDBfile_ligand}.pdb
trajin  {PDBfile_ligand}.pdb
strip   !:{self.LIGAND}
trajout {PDBfile_out}_lig.pdb
'''
    with open(f'{PDBfile_out}_CPPTraj_lig.in','w') as f: f.write(cpptraj)

    # Get the ESMfold pdb file and strip away all hydrogens
    cpptraj = f'''parm    {PDBfile}.pdb
trajin  {PDBfile}.pdb
strip   !@C,N,O,CA
trajout {PDBfile_out}_new_bb.pdb
'''
    with open(f'{PDBfile_out}_CPPTraj_new_bb.in','w') as f: f.write(cpptraj)

    # Align substrate and ESM prediction of scaffold without hydrogens
    cpptraj = f'''parm    {PDBfile_out}_old_bb.pdb
reference {PDBfile_out}_old_bb.pdb [ref]
trajin    {PDBfile_out}_new_bb.pdb
rmsd      @CA ref [ref]
trajout   {PDBfile_out}_aligned.pdb noter
'''
    
#     cpptraj = f'''parm    {PDBfile_ligand}.pdb [ref_parm]
# parm    {PDBfile}.pdb [struc_parm]
# reference {PDBfile_ligand}.pdb parm [ref_parm] [ref]
# trajin    {PDBfile}.pdb parm [struc_parm]
# rmsd      @CA ref [ref]
# trajout   {PDBfile_out}_aligned.pdb parm [struc_parm] noter
# '''
    with open(f'{PDBfile_out}_CPPTraj_aligned.in','w') as f: f.write(cpptraj) 
 
    cmd += f"""
    
cpptraj -i {PDBfile_out}_CPPTraj_old_bb.in &> \
           {PDBfile_out}_CPPTraj_old_bb.out
cpptraj -i {PDBfile_out}_CPPTraj_lig.in &> \
           {PDBfile_out}_CPPTraj_lig.out
cpptraj -i {PDBfile_out}_CPPTraj_new_bb.in &> \
           {PDBfile_out}_CPPTraj_new_bb.out
cpptraj -i {PDBfile_out}_CPPTraj_aligned.in &> \
           {PDBfile_out}_CPPTraj_aligned.out

# Cleanup structures
sed -i '/END/d' {PDBfile_out}_aligned.pdb
sed -i -e 's/^ATOM  /HETATM/' -e '/^TER/d' {PDBfile_out}_lig.pdb
    """
        
    remark = generate_remark_from_all_scores_df(self, index)
    with open(f'{PDBfile_out}_input.pdb', 'w') as f: f.write(remark+"\n")
    cmd += f"""# Assemble structure
cat {PDBfile_out}_aligned.pdb >> {PDBfile_out}_input.pdb
cat {PDBfile_out}_lig.pdb     >> {PDBfile_out}_input.pdb
sed -i '/TER/d' {PDBfile_out}_input.pdb
    """

#Write Rosetta input to build in sidechains
    Rosetta_xml = f"""
<ROSETTASCRIPTS>
    <MOVERS>
        <DumpPdb name="output_pdb" fname="{PDBfile_out}_wRosetta_sidechains.pdb"/>
    </MOVERS>
    <PROTOCOLS>
        <Add mover_name="output_pdb" />
    </PROTOCOLS>
</ROSETTASCRIPTS>
"""
    with open(f'{filename}/scripts/Rosetta_buildPDB_{index}.xml', 'w') as f:
        f.write(Rosetta_xml)

#Add Rosetta command to shell script
    cmd += f"""
# Run Rosetta Relax
{self.ROSETTA_PATH}/bin/rosetta_scripts.{self.rosetta_ext} \
                -s                        {PDBfile_out}_input.pdb \
                -extra_res_fa             {self.FOLDER_INPUT}/{self.LIGAND}.params \
                -parser:protocol          {filename}/scripts/Rosetta_buildPDB_{index}.xml \
                -nstruct                  1 \
                -ignore_zero_occupancy    false \
                -corrections::beta_nov16  true \
                -run:preserve_header      true \
                    
# Tidy up the output file
sed -i -e '/[0-9]H/d' -e '/H[0-9]/d' -e '/CONECT/Q' {PDBfile_out}_wRosetta_sidechains.pdb
"""

    with open(f"{PDBfile_out}_tleap.in", "w") as f:
        f.write(f"""source leaprc.protein.ff19SB 
source leaprc.gaff
loadamberprep   {self.FOLDER_INPUT}/5TS.prepi
loadamberparams {self.FOLDER_INPUT}/5TS.frcmod
mol = loadpdb {PDBfile_out}_wRosetta_sidechains.pdb
set default pbradii mbondi3
saveamberparm mol {PDBfile_out}.parm7 {PDBfile_out}.rst7
quit
""")
        
    cmd += f"""# Generate AMBER input files
tleap -s -f {PDBfile_out}_tleap.in &> {PDBfile_out}_tleap.out
    """

    # Create the input file    
    in_file = f"""
initial minimization
&cntrl                                                                         
    imin           = 1,                                                        
    ntmin          = 0,                                                        
    ncyc           = 500,                                                      
    maxcyc         = 1000,                                                      
    ntpr           = 100,                                                                                                                   
    ntb=0                                                                      
    igb = 8                                                                    
    cut = 1000,                                                                                                                   
&end 
"""    
    # Write the Amber.in to a file
    with open(f'{filename}/scripts/MDmin_{index}.in', 'w') as f:
        f.writelines(in_file)  

    cmd += f"""
# Run MD
sander -O -i {filename}/scripts/MDmin_{index}.in \
          -c {PDBfile_out}.rst7 -p {PDBfile_out}.parm7 \
          -o {PDBfile_out}_MD.log -x {PDBfile_out}_MD.nc -r {PDBfile_out}_MD_out.rst7 -inf {PDBfile_out}_MD_out.mdinf

cpptraj -p {PDBfile_out}.parm7 -y {PDBfile_out}_MD_out.rst7 -x {self.WT}_MDMin_{index}.pdb

# Clean the output file
sed -i '/        H  /d' {self.WT}_MDMin_{index}.pdb
    """

    return cmd