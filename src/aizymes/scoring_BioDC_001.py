"""
Calculates Redox potential using the BioDC software.

Functions:
    - prepare_BioDC: Prepares commands for BioDC job submission.
"""

import os
import shutil
import numpy as np

def prepare_BioDC(self, 
                       index, 
                       cmd):
    """
    Redox potential calculation for {index} with BioDC
    
    Parameters:
    index (str): The index of the protein variant to be analyzed.
    cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
    cmd (str): Command to be exected by run_design using submit_job.
    """

    
    filename = f'{self.FOLDER_DESIGN}/{index}'
    filename_in = self.all_scores_df.at[index, "final_variant"]
    variant_name = os.path.basename(filename_in)
    # Make directories
    os.makedirs(f"{filename}/scripts", exist_ok=True)
    os.makedirs(f"{filename}/BioDC", exist_ok=True)
    os.makedirs(f"{filename}/BioDC/{variant_name}", exist_ok=True)

    with open(f"{filename_in}.pdb", "r") as f:
        lines = f.readlines()
    
    with open(f"{filename}/BioDC/{variant_name}/{variant_name}.pdb", "w") as f:
        for line in lines:
            if not (line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER")): continue
            if "OXT" in line: continue
            if line.split()[-1] == "H": continue
            if "HEM X   1" in line:
                line = line.replace("HEM X   1", f"HEM B {int(self.HEME_RESI[2])}")
                line = line.replace("HETATM", "ATOM  ")
            f.write(line)
            
    input_file = f"""DivSel = 1                     
OriginalPDB = {variant_name}
SelDisulfides = no
ChooseMut = no
SelCpH = yes
SelectAllTitratable = yes
CreateResIndexingMethod = auto
ProcessPDBChoice = yes
RedoxState_1 = R
OutPrefix = BioDC
SolvEnv = imp
StructRelaxCompChoice = S
NProc = 1
continue_to_div_2 = yes
use_existing_min = yes
confirm_single_heme = yes
calculator_selection = 1
protein_dielectric = 5.5 
solvent_dielectric = 78.2
dg_method = 1
ref_state = red
use_eps_113 = yes
pbsa_type = 1
use_membrane = no
run_parallel = no
istrng_113 = 150
continue_calculations = no
continue_to_div_3 = no
"""
    with open(f"{filename}/BioDC/{variant_name}/input.txt", "w") as f:
        f.write(input_file)

    cmd += f"""### BioDC ####

cd BioDC/{variant_name}

module load vmd/1.9.3 amber/24 gcc/12 impi/2021.9

echo "Starting Xvfb..."
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99
export VMDFORCECPU=1 
sleep 2
biodc-cli
kill %1

cd ../..
"""
    
    if self.REMOVE_TMP_FILES:
        cmd += f"""
# Removing temporary directory
rm -r BioDC
"""
        
    return cmd

def get_redox_score(self, index, score_type):
  
    with open(f"{self.FOLDER_DESIGN}/{int(index)}/BioDC/{self.WT}_{score_type}_{index}/EE/DG.txt", "r") as f:
        redoxpotential = f.read()
    redoxpotential = float(redoxpotential.split()[-2])
    #redox_score = (self.TARGET_REDOX-redoxpotential)**2 # previously used. likely junk. makred for deletion by HAB
    redox_score = np.log(abs(self.TARGET_REDOX-redoxpotential))
    
    return redox_score, redoxpotential

def BioDC_check(self):
    """
    Checks if BioDC is installed
    """
    if not os.path.exists(self.FOLDER_BioDC):
        logging.error(f"BioDC not installed in {self.FOLDER_BioDC}.")
        logging.error(f"Install using: 'cd {os.path.dirname(self.FOLDER_BioDC)} && \
        git clone -b dev_v3.2 https://github.com/Mag14011/BioDC.git && \
        cd BioDC && \
        pip uninstall biodc  && \
        pip install . ")
        sys.exit()