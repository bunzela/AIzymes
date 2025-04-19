"""
Prepares commands to run protein structure prediction using AlphaFold3.

Functions:
    seq_to_json():              Converts a sequence file into a JSON input for AlphaFold3.
    prepare_AlphaFold3_MSA():   Prepares the AlphaFold3 MSA stage.
    prepare_AlphaFold3_INF():   Prepares the AlphaFold3 inference stage.
"""

import os
import time

def seq_to_json(self, seq_input, working_dir):
    
    """
    Converts a sequence file (with a .seq extension) into a JSON input file for AlphaFold3.

    Parameters:
        seq_input (str):    Input sequence file location
        working_dir (str):  working dir for AlphaFold3
    """

    for attempt in range(10): 
        try:
            with open(f'{seq_input}.seq', "r") as f:
                sequence = f.read()
            break  
        except IOError:
            time.sleep(1)
    else:
        raise IOError(f"Failed to read {file_path} after 10 attempts.")

    # Checks if AlphaFold3MSA is run with or without MSA
    if 'AlphaFold3MSA' in self.DESIGN_METHODS:
        MSA = ""
    else:
        MSA = """,
        "unpairedMsa": "",
        "pairedMsa": "",
        "templates": []
"""
      
    json = f'''
    {{
  "name": "{self.WT}",
  "sequences": [
    {{
      "protein": {{
        "id": ["A"],
        "sequence": "{sequence}"{MSA}
      }}
    }},
    {{
      "ligand": {{
        "id": ["X"],
        "ccdCodes": ["{self.LIGAND}"]
      }}
    }}
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}}
'''
    
    with open(f'{working_dir}.json', "w") as f: 
        f.write(json)

def prepare_AlphaFold3_MSA(self, 
                           index, 
                           cmd):
    """
    Performs MSA for structure prediction with AlphaFold3.
    
    Parameters:
        index (str):    The index of the protein variant to be predicted.
        cmd (str):      Growing list of commands to be exected by run_design using submit_job.

    Returns:
        cmd (str):      Command to be exected by run_design using submit_job.
    """
    
    working_dir = f'{self.FOLDER_DESIGN}/{index}'

    # Find input sequence !!!! MIGHT NOT BE CORRECT !!!
    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"]
    seq_input = f"{working_dir}/{self.WT}_{index}"
    
    # Make directories
    os.makedirs(f"{working_dir}/scripts", exist_ok=True)
    os.makedirs(f"{working_dir}/AlphaFold3", exist_ok=True)

    # Update working dir
    working_dir = f"{working_dir}/AlphaFold3"

    # Create json from sequence
    seq_to_json(self, seq_input, f"{working_dir}/{self.WT}_{index}")
    
    cmd += f"""### AlphaFold3_MSA ###

# Modules to load
module purge
module load apptainer/1.3.2
module load alphafold/3.0.1

# AlphaFold3 specific variables
AF3_MODEL_DIR={self.FOLDER_Alphafold}
AF3_JAX_CACHE_DIR={working_dir}/jaxcache

# Model specific variables
AF3_MSA_OUTPUT_DIR={working_dir}/MSA
AF3_MSA_JSON_PATH={working_dir}/{self.WT}_{index}.json

mkdir -p $AF3_JAX_CACHE_DIR $AF3_INFERENCE_OUTPUT_DIR

export OMP_NUM_THREADS=1

apptainer --quiet exec --bind /u:/u,/ptmp:/ptmp,/raven:/raven \
    ${{AF3_IMAGE_SIF}} \
    python3 /app/alphafold/run_alphafold.py \
    --db_dir $AF3_DB_DIR \
    --model_dir $AF3_MODEL_DIR \
    --json_path $AF3_MSA_JSON_PATH \
    --output_dir $AF3_MSA_OUTPUT_DIR \
    --norun_inference
"""
    return cmd
    
def prepare_AlphaFold3_INF(self, 
                           index, 
                           cmd,
                           gpu_id = None):
    """
    Performs MSA for structure prediction witt AlphaFold3.
    
    Parameters:
        index (str):    The index of the protein variant to be predicted.
        cmd (str):      Growing list of commands to be exected by run_design using submit_job.

    Returns:
        cmd (str):      Command to be exected by run_design using submit_job.
    """
    working_dir = f'{self.FOLDER_DESIGN}/{index}'

    # Find input sequence !!!! MIGHT NOT BE CORRECT !!!
    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"]
    seq_input = f"{working_dir}/{self.WT}_{index}"
            
    # Make directories
    os.makedirs(f"{working_dir}/scripts", exist_ok=True)
    os.makedirs(f"{working_dir}/AlphaFold3", exist_ok=True)

    # Update working dir
    working_dir = f"{working_dir}/AlphaFold3"

    # Create json from sequence
    seq_to_json(self, seq_input, f"{working_dir}/{self.WT}_{index}")
    
    cmd += f"""### AlphaFold3_MSA ###
"""
    # Set GPU
    if  gpu_id != None:
        cmd += f"""
export CUDA_VISIBLE_DEVICES={gpu_id}
"""

    # Checks if AlphaFold3MSA is run with or without MSA
    if 'AlphaFold3MSA' in self.DESIGN_METHODS:
        MSA = f"{working_dir}/MSA/{self.WT}/{self.WT}_data.json"
    else:
        MSA = f"{working_dir}/{self.WT}_{index}.json"
       
    cmd += f"""
# Modules to load
module purge
module load apptainer/1.3.2
module load alphafold/3.0.1

# AlphaFold3 specific variables
AF3_MODEL_DIR={self.FOLDER_Alphafold}
AF3_JAX_CACHE_DIR={working_dir}/jaxcache

# Model specific variables  
AF3_INFERENCE_OUTPUT_DIR={working_dir}/INF
AF3_INFERENCE_JSON_PATH={MSA}

mkdir -p $AF3_JAX_CACHE_DIR $AF3_INFERENCE_OUTPUT_DIR

export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION="3.0"

apptainer --quiet exec --bind /u:/u,/ptmp:/ptmp,/raven:/raven --nv \
    ${{AF3_IMAGE_SIF}} \
    python3 /app/alphafold/run_alphafold.py \
    --db_dir $AF3_DB_DIR \
    --model_dir $AF3_MODEL_DIR \
    --json_path $AF3_INFERENCE_JSON_PATH \
    --output_dir $AF3_INFERENCE_OUTPUT_DIR \
    --jax_compilation_cache_dir $AF3_JAX_CACHE_DIR \
    --norun_data_pipeline

{self.bash_args}python {self.FOLDER_PARENT}/cif_to_pdb.py \
--cif_file {working_dir}/INF/{self.WT}/{self.WT}_model.cif \
--pdb_file {working_dir}/../{self.WT}_AlphaFold3INF_{index}.pdb
"""
    
    if self.REMOVE_TMP_FILES:
        cmd += f"""# Removing temporary directory
rm -r AlphaFold3
"""

    return cmd   