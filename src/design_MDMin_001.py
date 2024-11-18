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
    
    input_pdb_paths = get_PDB_in(self, index)
    PDBfile_in = input_pdb_paths['MDMin_in']
        
    working_dir_path = f"{filename}/MDMin/{self.WT}_{index}"

    cmd += f"""# Assemble structure
cat {PDBfile_in}.pdb > {working_dir_path}_input.pdb
"""
    

#Write Rosetta input to build in sidechains
    Rosetta_xml = f"""
<ROSETTASCRIPTS>
    <MOVERS>
        <DumpPdb name="output_pdb" fname="{working_dir_path}_wRosetta_sidechains.pdb"/>
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
                -s                        {working_dir_path}_input.pdb \
                -extra_res_fa             {self.FOLDER_INPUT}/{self.LIGAND}.params \
                -parser:protocol          {filename}/scripts/Rosetta_buildPDB_{index}.xml \
                -nstruct                  1 \
                -ignore_zero_occupancy    false \
                -corrections::beta_nov16  true \
                -run:preserve_header      true \
                    
# Tidy up the output file
sed -i -e '/[0-9]H/d' -e '/H[0-9]/d' -e '/CONECT/Q' {working_dir_path}_wRosetta_sidechains.pdb

#Remove Rosetta output 
rm {self.WT}_{index}_input_0001.pdb
"""

    with open(f"{working_dir_path}_tleap.in", "w") as f:
        f.write(f"""source leaprc.protein.ff19SB 
source leaprc.gaff
loadamberprep   {self.FOLDER_INPUT}/5TS.prepi
loadamberparams {self.FOLDER_INPUT}/5TS.frcmod
mol = loadpdb {working_dir_path}_wRosetta_sidechains.pdb
set default pbradii mbondi3
saveamberparm mol {working_dir_path}.parm7 {working_dir_path}.rst7
quit
""")
        
    cmd += f"""# Generate AMBER input files
tleap -s -f {working_dir_path}_tleap.in &> {working_dir_path}_tleap.out
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
echo Starting MD
sander -O -i {filename}/scripts/MDmin_{index}.in \
          -c {working_dir_path}.rst7 -p {working_dir_path}.parm7 \
          -o {working_dir_path}_MD.log -x {working_dir_path}_MD.nc -r {working_dir_path}_MD_out.rst7 -inf {working_dir_path}_MD_out.mdinf

cpptraj -p {working_dir_path}.parm7 -y {working_dir_path}_MD_out.rst7 -x {working_dir_path}_MD_out.pdb

# Clean the output file
sed -i '/        H  /d' {working_dir_path}_MD_out.pdb
    """
    remark = generate_remark_from_all_scores_df(self, index)

    cmd += f"""
echo '{remark}' > {self.WT}_MDMin_{index}.pdb
cat {working_dir_path}_MD_out.pdb >> {self.WT}_MDMin_{index}.pdb
sed -i -e 's/^\(ATOM.\{{17\}}\) /\\1A/' {self.WT}_MDMin_{index}.pdb
sed -i -e 's/5TS A/5TS X/' {self.WT}_MDMin_{index}.pdb
    """

    return cmd