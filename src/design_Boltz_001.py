"""
Prepares commands to run protein structure prediction using Boltz-1.

Functions:
    - seq_to_yaml: Converts a sequence file into a YAML input for Boltz-1.
    - prepare_Boltz_INF: Prepares the Boltz inference stage for structure prediction.
"""

import os
import time
import logging

def seq_to_yaml(self, seq_input, working_dir, yaml_type="basic"):
    """
    Converts a sequence file into a YAML input for Boltz-1.
    
    Parameters:
    seq_input (str): Path to the sequence input file without extension
    working_dir (str): Working directory for the output yaml
    yaml_type (str): Type of YAML to generate (basic or glue)
    
    Returns:
    None: Writes the YAML file to the specified location
    """
    
    for attempt in range(10): 
        try:
            with open(f'{seq_input}.seq', "r") as f:
                sequence = f.read()
            break  
        except IOError:
            time.sleep(1)
    else:
        raise IOError(f"Failed to read {seq_input}.seq after 10 attempts.")

    # Get ligand SMILES if available
    ligand_smiles = ""
    if hasattr(self, 'LIGAND_SMILES') and self.LIGAND_SMILES:
        ligand_smiles = self.LIGAND_SMILES
    
    # Create yaml content as a string to avoid yaml package dependency
    yaml_content = f"""version: 1
sequences:
- protein:
    id: A
    sequence: "{sequence}"
"""
    
    # Only set MSA to empty if not using MSA server
    if not self.BOLTZ_USE_MSA_SERVER:
        yaml_content += '    msa: "empty"\n'
    
    # Add ligand entry with the ligand ID matching AIzymes convention and proper SMILES
    yaml_content += f"""- ligand:
    id: X
"""
    
    # Add either SMILES or CCD code depending on what's available
    if ligand_smiles:
        yaml_content += f'    smiles: "{ligand_smiles}"\n'
    else:
        # Fall back to CCD code if SMILES not available
        yaml_content += f'    ccd: "{self.LIGAND}"\n'
    
    # Write YAML to file
    with open(f'{working_dir}.yaml', "w") as f:
        f.write(yaml_content)
    
def prepare_Boltz_INF(self, 
                     index, 
                     cmd,
                     gpu_id = None):
    """
    Prepares the inference stage for structure prediction with Boltz-1.
    
    Parameters:
    index (str): The index of the protein variant to be predicted.
    cmd (str): Growing list of commands to be executed by run_design using submit_job.
    gpu_id (str, optional): GPU ID to use for inference.

    Returns:
    cmd (str): Command to be executed by run_design using submit_job.
    """
    working_dir = f'{self.FOLDER_DESIGN}/{index}'

    # Find input sequence
    seq_input = f"{working_dir}/{self.WT}_{index}"
            
    # Make directories
    os.makedirs(f"{working_dir}/scripts", exist_ok=True)
    os.makedirs(f"{working_dir}/Boltz", exist_ok=True)

    # Update working dir
    working_dir = f"{working_dir}/Boltz"

    # Create yaml from sequence
    seq_to_yaml(self, seq_input, f"{working_dir}/{self.WT}_{index}")
    
    # Output path follows AIzymes naming convention
    output_path = f"{self.FOLDER_DESIGN}/{index}/{self.WT}_BoltzINF_{index}.pdb"
    
    cmd += f"""### BoltzINF ###

"""
    # Set GPU
    if gpu_id is not None:
        cmd += f"""export CUDA_VISIBLE_DEVICES={gpu_id}
"""
       
    cmd += f"""# Ensure Boltz environment is activated
{self.BOLTZ_ENV_ACTIVATE}

# Boltz specific variables  
BOLTZ_OUTPUT_DIR={working_dir}/INF
BOLTZ_YAML_PATH={working_dir}/{self.WT}_{index}.yaml

mkdir -p $BOLTZ_OUTPUT_DIR

# Run Boltz inference
boltz predict $BOLTZ_YAML_PATH \\
    --out_dir $BOLTZ_OUTPUT_DIR \\
    --recycling_steps {self.BOLTZ_RECYCLING_STEPS} \\
    --sampling_steps {self.BOLTZ_SAMPLING_STEPS} \\
    --accelerator {"gpu" if gpu_id is not None else "cpu"} \\
    --step_scale {self.BOLTZ_STEP_SCALE} \\
    --diffusion_samples {self.BOLTZ_DIFFUSION_SAMPLES} \\
    --output_format pdb"""

    # Add the MSA server option if enabled
    if self.BOLTZ_USE_MSA_SERVER:
        cmd += f" \\\n    --use_msa_server"

    cmd += f"""

# Process results using helper script
{self.bash_args}python {self.FOLDER_PARENT}/process_boltz_results.py \\
    --boltz_dir $BOLTZ_OUTPUT_DIR \\
    --output_pdb {output_path}
"""
    return cmd 