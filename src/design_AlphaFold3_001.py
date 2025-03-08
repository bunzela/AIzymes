"""
Design AlphaFold3 Module

Manages the structure prediction of protein sequences using AlphaFold3 within the AIzymes project.

Functions:
    - prepare_AlphaFold3: Prepares commands for AlphaFold3 job submission.
"""
import logging
import os
import shutil 
import subprocess  
import gemmi

from helper_002               import *

def seq_to_json(self, seq_input, working_dir):

    with open(f'{seq_input}.seq', "r") as f: 
        sequence=f.read()

    json = f'''
    {{
  "name": "{self.WT}",
  "sequences": [
    {{
      "protein": {{
        "id": ["A"],
        "sequence": "{sequence}"
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
    index (str): The index of the protein variant to be predicted.
    cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
    cmd (str): Command to be exected by run_design using submit_job.
    """
    working_dir = f'{self.FOLDER_HOME}/{index}'

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
AF3_USER_DIR=${{HOME}}/bin/alphafold_3_0_1
AF3_MODEL_DIR=${{AF3_USER_DIR}}/model
AF3_JAX_CACHE_DIR={working_dir}/jaxcache

# Model specific variables
AF3_MSA_OUTPUT_DIR={working_dir}/MSA
AF3_INFERENCE_OUTPUT_DIR={working_dir}/INF
AF3_MSA_JSON_PATH={working_dir}/{self.WT}_{index}.json
AF3_INFERENCE_JSON_PATH={working_dir}/MSA/{self.WT}/{self.WT}_data.json

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
                           cmd):
    """
    Performs MSA for structure prediction witt AlphaFold3.
    
    Parameters:
    index (str): The index of the protein variant to be predicted.
    cmd (str): Growing list of commands to be exected by run_design using submit_job.

    Returns:
    cmd (str): Command to be exected by run_design using submit_job.
    """
    working_dir = f'{self.FOLDER_HOME}/{index}'
        
    # Make directories
    os.makedirs(f"{working_dir}/scripts", exist_ok=True)
    os.makedirs(f"{working_dir}/AlphaFold3", exist_ok=True)

    working_dir = f"{working_dir}/AlphaFold3"

    cmd += f"""### AlphaFold3_MSA ###

# Modules to load
module purge
module load apptainer/1.3.2
module load alphafold/3.0.1

# AlphaFold3 specific variables
AF3_USER_DIR=${{HOME}}/bin/alphafold_3_0_1
AF3_MODEL_DIR=${{AF3_USER_DIR}}/model
AF3_JAX_CACHE_DIR={working_dir}/jaxcache

# Model specific variables  
AF3_MSA_OUTPUT_DIR={working_dir}/MSA
AF3_INFERENCE_OUTPUT_DIR={working_dir}/INF
AF3_MSA_JSON_PATH={working_dir}/{self.WT}_{index}.json
AF3_INFERENCE_JSON_PATH={working_dir}/MSA/{self.WT}/{self.WT}_data.json

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
--cif_file {working_dir}/INF/{self.WT}/{self.WT}/{self.WT}_model.cif \
--pdb_file {working_dir}/../{self.WT}_AlphaFold3INF_{index}.pdb
"""
    return cmd