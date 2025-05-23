"""
Prepares commands and input files to run Boltz-1 for protein-ligand prediction.

Functions:
    prepare_Boltz: Generates FASTA input and prepares Boltz-1 command for job submission.
"""

import os
import logging
import sys
import shlex # Use shlex to quote paths for shell safety
from Bio import SeqIO # Added for fallback sequence_from_pdb
import pandas as pd
from helper_002 import generate_remark_from_all_scores_df # <-- ADD IMPORT

# Using the built-in sequence_from_pdb logic directly
# Removed the try-except block for helper_002 as it was causing linting errors
# and a fallback was already defined.
def _sequence_from_pdb(pdb_in_path):
    """Extracts the first protein sequence from a PDB file."""
    # Ensure the file exists before opening
    if not os.path.exists(pdb_in_path):
         logging.error(f"PDB file {pdb_in_path} not found for sequence extraction.")
         # Return a default/placeholder or raise an error, depending on desired handling
         return "SEQUENCENOTFOUND" # Placeholder
    try:
        with open(pdb_in_path, "r") as f:
            for record in SeqIO.parse(f, "pdb-atom"):
                # Return the sequence of the first model/record found
                return str(record.seq)
        # Handle case where PDB is valid but contains no parseable sequence
        logging.warning(f"Could not parse sequence from PDB: {pdb_in_path}")
        return "SEQUENCENOTPARSED"
    except Exception as e:
         logging.error(f"Error parsing PDB file {pdb_in_path}: {e}")
         return "SEQUENCEPARSEERROR"

def _generate_boltz_fasta(self, index, fasta_path):
    """
    Generates the FASTA input file for Boltz-1 based on AIzymes config.

    Handles sequence retrieval from .seq file or fallback to .pdb.
    Requires BOLTZ_LIGAND_TYPE ('smiles' or 'ccd') and BOLTZ_LIGAND_IDENTIFIER in self.
    
    When BOLTZ_LIGAND_TYPE is 'smiles', BOLTZ_LIGAND_IDENTIFIER should be a valid SMILES string.
    When BOLTZ_LIGAND_TYPE is 'ccd', BOLTZ_LIGAND_IDENTIFIER should be a valid CCD identifier.
    """
    logging.debug(f"Generating Boltz FASTA for index {index} at {fasta_path}")

    # --- Get Protein Sequence ---
    # step_input_variant usually doesn't have the extension
    protein_input_base = self.all_scores_df.at[int(index), "step_input_variant"]
    protein_seq_file = f"{protein_input_base}.seq"
    protein_pdb_file = f"{protein_input_base}.pdb" # Keep the .pdb extension

    protein_sequence = None
    if os.path.exists(protein_seq_file):
        logging.debug(f"Reading sequence from {protein_seq_file}")
        with open(protein_seq_file, 'r') as f:
            protein_sequence = f.read().strip()
    elif os.path.exists(protein_pdb_file):
        # Fallback: Try extracting sequence from the PDB if .seq file is missing
        logging.warning(f"Sequence file {protein_seq_file} not found. Attempting extraction from {protein_pdb_file}.")
        protein_sequence = _sequence_from_pdb(protein_pdb_file) # Pass the full path
    else:
        logging.error(f"Cannot find sequence source: Neither {protein_seq_file} nor {protein_pdb_file} exist for Boltz index {index}.")
        # Return False or raise an error to indicate failure
        return False

    # Check if sequence retrieval failed
    if not protein_sequence or "NOTFOUND" in protein_sequence or "ERROR" in protein_sequence or "NOTPARSED" in protein_sequence:
        logging.error(f"Failed to obtain a valid protein sequence for Boltz index {index}. Sequence found: '{protein_sequence}'")
        return False

    # --- Get Ligand Info ---
    try:
        # Default ligand_type to "smiles" instead of "ccd" for better compatibility with diverse ligands
        ligand_type = getattr(self, 'BOLTZ_LIGAND_TYPE', 'smiles')
        if not ligand_type:
            ligand_type = 'smiles'  # Default to SMILES format if not specified
        ligand_type = ligand_type.lower() # Always convert to lowercase
        
        # If BOLTZ_LIGAND_IDENTIFIER is None, use the value from LIGAND
        ligand_id = self.BOLTZ_LIGAND_IDENTIFIER if self.BOLTZ_LIGAND_IDENTIFIER is not None else self.LIGAND
        if ligand_id is None:
            logging.error(f"Boltz index {index}: Neither BOLTZ_LIGAND_IDENTIFIER nor LIGAND is set")
            return False # Indicate failure
            
        # Validate ligand based on type
        if ligand_type == 'smiles':
            if not isinstance(ligand_id, str) or not ligand_id.strip():
                logging.error(f"Boltz index {index}: Invalid SMILES string: '{ligand_id}'")
                return False
            logging.info(f"Boltz index {index}: Using SMILES string for ligand: '{ligand_id}'")
        elif ligand_type == 'ccd':
            logging.info(f"Boltz index {index}: Using CCD identifier for ligand: '{ligand_id}'")
        else:
            # For other ligand types, just log the information
            logging.info(f"Boltz index {index}: Using ligand type '{ligand_type}' with identifier '{ligand_id}'")
    except AttributeError as e:
        logging.error(f"Missing required Boltz ligand attribute in setup for index {index}: {e}. Exiting.")
        return False # Indicate failure

    # --- Write FASTA ---
    # Chain A: Protein, Chain B: Ligand
    try:
        with open(fasta_path, 'w') as f:
            # According to Boltz docs, format is >CHAIN_ID|ENTITY_TYPE|MSA_PATH
            # For single sequence mode, use >A|protein|empty 
            # Check if using MSA server or not
            if getattr(self, 'BOLTZ_USE_MSA_SERVER', False):
                f.write(">A|protein\n") # When using MSA server, no need to specify MSA path
            else:
                f.write(">A|protein|empty\n") # For single sequence mode, specify "empty"
            
            f.write(f"{protein_sequence}\n") # Protein sequence
            f.write(f">B|{ligand_type}\n") # Header for ligand
            f.write(f"{ligand_id}\n") # Ligand identifier
        logging.info(f"Successfully generated Boltz FASTA input: {fasta_path}")
        return True # Indicate success
    except IOError as e:
        logging.error(f"Failed to write Boltz FASTA file {fasta_path}: {e}")
        return False # Indicate failure


def prepare_Boltz(self, index, cmd, gpu_id=None):
    """
    Prepares the command to run Boltz-1 prediction for the variant specified by index.
    Follows AIzymes conventions for tool integration.

    Args:
        self: The AIzymes_MAIN instance (contains configuration).
        index (int): The variant index.
        cmd (str): The accumulating command string for the job script.
        gpu_id (int, optional): The assigned GPU ID, if any. Defaults to None.

    Returns:
        str: The updated command string including Boltz-1 prediction step.
             Returns original cmd + error comment if setup fails.
    """
    logging.info(f"Preparing Boltz-1 for index {index}.")

    # --- Check for required Boltz parameters ---
    required_attrs = ['FOLDER_DESIGN', 'WT', 'all_scores_df']
    missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr) or getattr(self, attr) is None]
    if missing_attrs:
        logging.error(f"Missing required Boltz attributes in setup for index {index}: {', '.join(missing_attrs)}")
        return cmd + f"\n# ERROR: Missing Boltz attributes: {', '.join(missing_attrs)}\n"

    # --- Define Paths (Using AIzymes conventions) ---
    try:
        index_str = str(index)
        folder_design = getattr(self, 'FOLDER_DESIGN') # Assume FOLDER_DESIGN is always set
        wt_name = getattr(self, 'WT') # Assume WT is always set

        # Consistent working directory structure
        boltz_work_dir_py = os.path.join(folder_design, index_str, "Boltz")
        # Standard AIzymes output directory for this index
        aizymes_index_dir_py = os.path.join(folder_design, index_str)

        # Ensure necessary directories exist (create them using Python)
        os.makedirs(boltz_work_dir_py, exist_ok=True)
        # No need to create aizymes_index_dir_py here, assumed to exist

        # Input FASTA path
        fasta_input_path_py = os.path.join(boltz_work_dir_py, f"{wt_name}_Boltz_{index_str}.fasta")

        # Shell-quoted paths for the command string
        boltz_work_dir_sh = shlex.quote(boltz_work_dir_py)
        fasta_input_path_sh = shlex.quote(fasta_input_path_py)

    except AttributeError as e:
        logging.error(f"Missing critical path attribute (FOLDER_DESIGN or WT) for Boltz index {index}: {e}")
        return cmd + f"\n# ERROR: Path definition failed (missing FOLDER_DESIGN or WT) for Boltz index {index}: {e}\n"
    except Exception as e:
        logging.error(f"Error defining paths for Boltz index {index}: {e}")
        return cmd + f"\n# ERROR: Path definition failed for Boltz index {index}: {e}\n"

    # --- Generate FASTA Input using Python Function ---
    # This now happens before command generation
    fasta_success = _generate_boltz_fasta(self, index, fasta_input_path_py)
    if not fasta_success:
        logging.error(f"Failed to generate FASTA input for Boltz index {index}. Aborting command generation.")
        return cmd + f"\n# ERROR: Failed to generate Boltz FASTA input for index {index}.\n"
    
    # --- Retrieve Protein Sequence Length for Ligand Renumbering ---
    # Reuse the logic from _generate_boltz_fasta to get the sequence
    protein_sequence = None
    protein_input_base = self.all_scores_df.at[int(index), "step_input_variant"]
    protein_seq_file = f"{protein_input_base}.seq"
    protein_pdb_file = f"{protein_input_base}.pdb"
    if os.path.exists(protein_seq_file):
        with open(protein_seq_file, 'r') as f:
            protein_sequence = f.read().strip()
    elif os.path.exists(protein_pdb_file):
        protein_sequence = _sequence_from_pdb(protein_pdb_file)
    
    if not protein_sequence or "NOTFOUND" in protein_sequence or "ERROR" in protein_sequence or "NOTPARSED" in protein_sequence:
         logging.error(f"Failed to obtain protein sequence again for renumbering in Boltz index {index}. Cannot determine ligand residue number.")
         # Provide a fallback or error handling? For now, use a default like 1 or fail.
         # Let's use a placeholder that will likely cause issues downstream if sequence isn't found,
         # signaling the problem clearly.
         protein_seq_len = 0 # Indicate failure
         ligand_resi_py = 9999 # Use an obviously wrong number
    else:
         protein_seq_len = len(protein_sequence)
         ligand_resi_py = protein_seq_len + 1
         logging.info(f"Boltz index {index}: Protein sequence length = {protein_seq_len}. Ligand residue number set to {ligand_resi_py}.")


    # --- Construct Boltz Command ---
    current_cmd = f"\n### Boltz-1 Prediction ###\n"

    # Add activation command if specified
    boltz_env_activate = getattr(self, 'BOLTZ_ENV_ACTIVATE', None)
    if boltz_env_activate:
        current_cmd += f"# Activating Boltz environment\n{boltz_env_activate}\n"
        
    # GPU environment setup
    accelerator = "cpu"
    devices = 1 # Default to 1 device

    # Simple GPU check identical to ESMfold
    if gpu_id != None:
        current_cmd += f"""
export CUDA_VISIBLE_DEVICES={gpu_id}
# Diagnostic for GPU availability
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
"""
        accelerator = "gpu"
        logging.info(f"Boltz index {index} assigned to GPU: {gpu_id}")
    else:
        # CPU fallback
        try:
            max_jobs_val = getattr(self, 'MAX_JOBS', 1)
            devices = int(max_jobs_val)
            if devices < 1: devices = 1
            logging.info(f"Boltz index {index} using CPU. Devices: {devices}")
        except (ValueError, TypeError):
            logging.warning(f"Could not convert MAX_JOBS to int, defaulting to 1 CPU device for Boltz index {index}.")
            devices = 1

    # Get Boltz parameters from config, using defaults if not present
    # Use getattr for safe access with defaults
    boltz_recycling_steps = getattr(self, 'BOLTZ_RECYCLING_STEPS', 3)
    boltz_sampling_steps = getattr(self, 'BOLTZ_SAMPLING_STEPS', 200)
    # Ensure diffusion samples >= devices for CPU run
    boltz_diffusion_samples = getattr(self, 'BOLTZ_DIFFUSION_SAMPLES', 1)
    if accelerator == "cpu" and devices > boltz_diffusion_samples:
        logging.warning(f"Boltz index {index}: Number of CPU devices ({devices}) > diffusion samples ({boltz_diffusion_samples}). Setting devices = diffusion samples.")
        devices = boltz_diffusion_samples


    boltz_use_msa_server = getattr(self, 'BOLTZ_USE_MSA_SERVER', False)
    boltz_msa_server_url = getattr(self, 'BOLTZ_MSA_SERVER_URL', 'https://api.colabfold.com')
    boltz_msa_pairing_strategy = getattr(self, 'BOLTZ_MSA_PAIRING_STRATEGY', 'greedy')

    # Build the boltz command using shlex.join for safety
    boltz_predict_cmd_parts = [
        "boltz", "predict", f"{fasta_input_path_sh}",
        "--out_dir", f"{boltz_work_dir_sh}",
        "--output_format", "pdb",  # Always use PDB format for AIzymes compatibility
        "--recycling_steps", str(boltz_recycling_steps),
        "--sampling_steps", str(boltz_sampling_steps),
        "--diffusion_samples", str(boltz_diffusion_samples),
        "--accelerator", accelerator,
        "--devices", str(devices) # Use determined number of devices
    ]

    if boltz_use_msa_server:
        boltz_predict_cmd_parts.extend([
            "--use_msa_server",
            "--msa_server_url", f"{boltz_msa_server_url}",
            "--msa_pairing_strategy", f"{boltz_msa_pairing_strategy}"
        ])
    else:
        # If not using server, check if a custom MSA path is provided
        msa_path = getattr(self, 'BOLTZ_MSA_PATH', None)
        if msa_path and os.path.exists(msa_path):
             boltz_predict_cmd_parts.extend(["--msa_path", shlex.quote(msa_path)])
             logging.info(f"Boltz index {index}: Using custom MSA path: {msa_path}")
        else:
             # If no MSA path, we've already specified "empty" in the FASTA header
             # No need to add --msa_path flag here since it's encoded in the FASTA
             logging.info(f"Boltz index {index}: Using single sequence mode via FASTA header")
             # No additional flags needed for single sequence mode

    # Add optional boolean flags from config
    if getattr(self, 'BOLTZ_OVERRIDE', False):
        boltz_predict_cmd_parts.append("--override")
    if getattr(self, 'BOLTZ_WRITE_FULL_PAE', False):
        boltz_predict_cmd_parts.append("--write_full_pae")
    if getattr(self, 'BOLTZ_WRITE_FULL_PDE', False):
        boltz_predict_cmd_parts.append("--write_full_pde")

    # Add the main boltz command
    current_cmd += shlex.join(boltz_predict_cmd_parts) + "\n\n"
    
    # Add diagnostic command to help debug GPU issues
    if accelerator == "gpu":
        current_cmd += "# GPU diagnostic info\n"
        current_cmd += "echo \"CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES\"\n"
        current_cmd += "if command -v nvidia-smi &> /dev/null; then nvidia-smi -L || echo \"nvidia-smi unavailable\"; fi\n\n"

    # --- Define Expected Output Paths and Copy Command (AIzymes Standard) ---
    input_stem = os.path.splitext(os.path.basename(fasta_input_path_py))[0] # e.g., WT_Boltz_0
    
    # Boltz internal prediction folder path - update to match actual structure
    # The format is: Boltz/boltz_results_{input_stem}/predictions/{input_stem}
    boltz_results_dir = os.path.join(boltz_work_dir_py, f"boltz_results_{input_stem}")
    prediction_folder_py = os.path.join(boltz_results_dir, "predictions", input_stem)

    # Path to the top-ranked predicted structure (model_0) inside Boltz output directory
    expected_output_structure_py = os.path.join(prediction_folder_py, f"{input_stem}_model_0.pdb")
    expected_output_structure_sh = shlex.quote(expected_output_structure_py)

    # AIzymes standard output path for the final structure in FOLDER_DESIGN/index/
    # Use the standard AIzymes pattern: {WT}_{method}_{index}.pdb to match other folding methods
    final_structure_aizymes_name_py = os.path.join(aizymes_index_dir_py, f"{wt_name}_Boltz_{index_str}.pdb")
    final_structure_aizymes_name_sh = shlex.quote(final_structure_aizymes_name_py)

    # Path to the confidence scores JSON file inside boltz_run/predictions/
    expected_confidence_file_py = os.path.join(prediction_folder_py, f"confidence_{input_stem}_model_0.json")
    expected_confidence_file_sh = shlex.quote(expected_confidence_file_py)
    # AIzymes standard path for scores file in FOLDER_DESIGN/index/
    final_confidence_aizymes_name_py = os.path.join(aizymes_index_dir_py, f"scores_Boltz_{index_str}.json")
    final_confidence_aizymes_name_sh = shlex.quote(final_confidence_aizymes_name_py)

    # ---> Retrieve Ligand Name and Calculated Residue Number <---
    ligand_name_py = self.LIGAND if hasattr(self, 'LIGAND') and self.LIGAND else 'LIG'
    # ligand_resi_py is calculated above after getting protein sequence length

    # ---> Generate remarks using the helper function BEFORE copy_cmd <---
    remarks_py = ""
    try:
        # Use the helper function
        remarks_py = generate_remark_from_all_scores_df(
            self, # Pass the AIzymes_MAIN instance
            index=index
        )
        logging.info(f"Generated REMARKs for Boltz index {index}:")
        # Log first few lines of remarks for verification if not empty
        if remarks_py:
            for line in remarks_py.split('\n')[:3]:
                logging.info(f"  {line}")
        else:
            logging.info("  (No remarks generated - likely missing score data)")
    except Exception as e:
        logging.error(f"Error generating remarks for Boltz index {index}: {e}")
        remarks_py = "REMARK 999 Error generating remarks\n" # Fallback remark

    # ---> Define AWK script parts as Python variables <---
    awk_script_primary_py = '''
    BEGIN { 
        if (new_resi+0 != new_resi) {
            print "AWK Warning: Invalid LIGAND_RESI received (" new_resi "), defaulting to 9999." > "/dev/stderr"
            new_resi = 9999 
        }
    }
    ( ($1 == "ATOM" || $1 == "HETATM") && substr($0, 18, 3) == "LIG" && substr($0, 22, 1) == "B" && substr($0, 23, 4)+0 == 1 ) {
        serial = substr($0, 7, 5); atom_name = substr($0, 13, 4); alt_loc = substr($0, 17, 1);
        icode = substr($0, 27, 1); x = substr($0, 31, 8); y = substr($0, 39, 8); z = substr($0, 47, 8);
        occ = substr($0, 55, 6); temp = substr($0, 61, 6); element = substr($0, 77, 2); charge = substr($0, 79, 2);
        printf "ATOM  %5s %-4s%1s%3s %1s%4d%1s   %8s%8s%8s%6s%6s          %2s%2s\\n", \
               serial, atom_name, alt_loc, new_resn, new_chain, new_resi, icode, x, y, z, occ, temp, element, charge
        found = 1 
    }
    END { 
        exit !found 
    }
    '''

    awk_script_fallback_py = '''
    BEGIN { 
        if (new_resi+0 != new_resi) { new_resi = 9999 } 
    }
    ( ($1 == "ATOM" || $1 == "HETATM") && substr($0, 18, 3) == "LIG" ) {
        serial = substr($0, 7, 5); atom_name = substr($0, 13, 4); alt_loc = substr($0, 17, 1);
        icode = substr($0, 27, 1); x = substr($0, 31, 8); y = substr($0, 39, 8); z = substr($0, 47, 8);
        occ = substr($0, 55, 6); temp = substr($0, 61, 6); element = substr($0, 77, 2); charge = substr($0, 79, 2);
        printf "ATOM  %5s %-4s%1s%3s %1s%4d%1s   %8s%8s%8s%6s%6s          %2s%2s\\n", \
               serial, atom_name, alt_loc, new_resn, new_chain, new_resi, icode, x, y, z, occ, temp, element, charge
        found = 1 
    }
    END { exit !found }
    '''


    # --- Generate the copy/format command ---
    copy_cmd = f"""
# Copy top Boltz prediction and scores to AIzymes standard location
echo "Checking for Boltz output structure: {expected_output_structure_sh}"
if [ -f "{expected_output_structure_sh}" ] && [ -s "{expected_output_structure_sh}" ]; then
  echo "Boltz output found and is not empty. Processing PDB for ESMfold/Rosetta compatibility..."
  # Optional: Add head/tail logging back if needed for debugging
  # echo "--- Start of Raw Boltz PDB ({expected_output_structure_sh}) ---" ... 

  TMP_PDB_OUT=$(mktemp)

  # 1. Prepare Remarks and Ligand Name
  LIGAND_NAME=$(printf '%s' {shlex.quote(ligand_name_py)})
  LIGAND_RESI=$(printf '%s' {shlex.quote(str(ligand_resi_py))}) # Use calculated residue number
  REMARKS=$(printf '%s' {shlex.quote(remarks_py)}) # remarks_py generated above in Python

  # 2. Write Remarks to temp file
  echo "REMARK   0 File processed from Boltz prediction" > "$TMP_PDB_OUT"
  echo "REMARK   0 Chain A: Protein, Chain X: Ligand ($LIGAND_NAME)" >> "$TMP_PDB_OUT"
  # Write the potentially multi-line REMARK 666 section
  echo "$REMARKS" >> "$TMP_PDB_OUT"

  # 3. Append Protein ATOM lines (Chain A)
  grep '^ATOM  ' "{expected_output_structure_sh}" | grep -v " $LIGAND_NAME " >> "$TMP_PDB_OUT" || echo "Warning: No protein ATOM lines found or grep failed."

  # 4. Append first TER
  echo "TER" >> "$TMP_PDB_OUT"

  # 5. Process and Append modified Ligand lines (Check HETATM then ATOM, apply transformations)
  LIGAND_LINES=$(mktemp)
  LIGAND_FOUND=0 # Flag to track if ligand was found

  # Use awk to find EITHER ATOM or HETATM lines for residue "LIG", chain "B", resi "1"
  # and transform them to ATOM, $LIGAND_NAME, chain "X", resi $LIGAND_RESI
  awk -v new_resn="$LIGAND_NAME" -v new_chain="X" -v new_resi="$LIGAND_RESI" \
    '{awk_script_primary_py}' \
    "{expected_output_structure_sh}" > "$LIGAND_LINES"

  # Check awk exit code and if the output file has content
  if [ $? -eq 0 ] && [ -s "$LIGAND_LINES" ]; then
    echo "Ligand lines found (Residue Name LIG) and processed successfully."
    cat "$LIGAND_LINES" >> "$TMP_PDB_OUT"
    LIGAND_FOUND=1
  else
    # Try again without chain/resi check, just in case Boltz format changes
    echo "Warning: Initial ligand search (ATOM/HETATM, LIG, B, 1) failed. Trying broader search (ATOM/HETATM, LIG)..." >&2
    awk -v new_resn="$LIGAND_NAME" -v new_chain="X" -v new_resi="$LIGAND_RESI" \
      '{awk_script_fallback_py}' \
      "{expected_output_structure_sh}" > "$LIGAND_LINES"

      if [ $? -eq 0 ] && [ -s "$LIGAND_LINES" ]; then
          echo "Ligand lines found on second attempt (Residue Name LIG, any chain/resi) and processed successfully."
          cat "$LIGAND_LINES" >> "$TMP_PDB_OUT"
          LIGAND_FOUND=1
      fi
  fi
  
  # Final check if ligand was processed
  rm -f "$LIGAND_LINES" # Clean up temp file regardless
  if [ "$LIGAND_FOUND" -eq 0 ]; then
    echo "ERROR: Failed to find or process ligand lines (Residue Name LIG) in {expected_output_structure_sh}. Raw file might be missing ligand or format is unexpected." >&2
    rm -f "$TMP_PDB_OUT"
    exit 1
  fi

  # 6. Append second TER
  echo "TER" >> "$TMP_PDB_OUT"

  # 7. Append END
  echo "END" >> "$TMP_PDB_OUT"

  # 8. Check and move the final file
  echo "grep/awk processing finished. Checking temporary output file: $TMP_PDB_OUT"
  if [ -s "$TMP_PDB_OUT" ]; then
    TER_COUNT=$(grep -c "^TER" "$TMP_PDB_OUT")
    if [ "$TER_COUNT" -eq 2 ]; then
      echo "Temporary output $TMP_PDB_OUT seems valid (size > 0, 2 TER records). Moving to final destination: {final_structure_aizymes_name_sh}"
      mv "$TMP_PDB_OUT" "{final_structure_aizymes_name_sh}"
      echo "PDB file processed with grep/awk and saved to {final_structure_aizymes_name_sh}"
    else
      echo "ERROR: Processed PDB file ($TMP_PDB_OUT) has $TER_COUNT TER records (expected 2). Check processing steps." >&2
      rm -f "$TMP_PDB_OUT"
      exit 1
    fi
  else
    echo "ERROR: grep/awk processing resulted in an empty temporary file ($TMP_PDB_OUT)." >&2
    rm -f "$TMP_PDB_OUT"
    exit 1
  fi

elif [ -f "{expected_output_structure_sh}" ]; then
    echo "ERROR: Expected Boltz output structure exists but is EMPTY: {expected_output_structure_sh}" >&2
    exit 1
else
  # Escape quotes in error message
  echo "ERROR: Expected Boltz output structure not found at: {expected_output_structure_sh}" >&2
  echo "Checking Boltz directories..."
  # Escape quotes in diagnostic messages
  echo "Contents of boltz_work_dir ({boltz_work_dir_sh}):"
  ls -la "{boltz_work_dir_sh}" || echo "Cannot access directory"
  if [ -d "{boltz_results_dir}" ]; then # Escape quotes around variable in test
    echo "Contents of boltz_results_dir:"
    ls -la {shlex.quote(boltz_results_dir)} || echo "Cannot access directory"
    predictions_dir={shlex.quote(os.path.join(boltz_results_dir, "predictions"))}
    if [ -d "$predictions_dir" ]; then
      echo "Contents of predictions directory:"
      ls -la "$predictions_dir" || echo "Cannot access directory"
      echo "Searching for any model files:"
      # Escape quotes for find command pattern
      find {shlex.quote(boltz_results_dir)} -name "*model_0*" -type f || echo "No model files found"
    fi
  fi
  echo "CRITICAL ERROR: Boltz prediction file not found. Check logs for details." >&2
  exit 1
fi

# Check for confidence file - escape quotes
echo "Checking for Boltz confidence file: {expected_confidence_file_sh}"
if [ -f "{expected_confidence_file_sh}" ]; then
  echo "Copying Boltz scores to {final_confidence_aizymes_name_sh}"
  cp "{expected_confidence_file_sh}" "{final_confidence_aizymes_name_sh}"
else
  echo "Warning: Boltz confidence file not found at: {expected_confidence_file_sh}" >&2
fi
"""
    current_cmd += copy_cmd.strip() + "\n"

    # --- Optional Cleanup ---
    if getattr(self, 'REMOVE_TMP_FILES', False):
         current_cmd += f"""
# Removing temporary Boltz directory
echo "Removing temporary directory: {boltz_work_dir_sh}"
rm -r {boltz_work_dir_sh}
"""

    logging.info(f"Boltz-1 command for index {index} prepared successfully.")
    # Append the newly generated commands to the original cmd string passed in
    return cmd + current_cmd.strip() + "\n" # Ensure single newline at the end


