import os
import logging
import sys
import shlex
import shutil

# Import helper functions (ensure helper_002 is in PYTHONPATH)
from helper_002 import sequence_from_pdb as _sequence_from_pdb, generate_remark_from_all_scores_df  # type: ignore


def _generate_chai1_fasta(self, index, fasta_path):
    """
    Generates the FASTA input file for Chai-1 based on AIzymes config.
    Handles single protein sequence retrieval from a .seq file, with a fallback to .pdb if needed.
    Expects self.LIGAND to be a SMILES string.
    Returns a tuple: (success: bool, protein_sequence_length: int or None)
    """
    logging.debug(f"Generating Chai-1 FASTA for index {index} at {fasta_path}")

    # Get protein input base from the scores DataFrame
    protein_input_base = self.all_scores_df.at[int(index), "step_input_variant"]
    protein_seq_file = f"{protein_input_base}.seq"
    protein_pdb_file = f"{protein_input_base}.pdb"

    protein_sequence = None
    if os.path.exists(protein_seq_file):
        logging.debug(f"Reading sequence from {protein_seq_file}")
        try:
            with open(protein_seq_file, 'r') as f:
                protein_sequence = f.read().strip()
        except Exception as e:
            logging.error(f"Error reading {protein_seq_file}: {e}")
    elif os.path.exists(protein_pdb_file):
        logging.warning(f"Sequence file {protein_seq_file} not found. Extracting from {protein_pdb_file}.")
        protein_sequence = _sequence_from_pdb(protein_input_base)
    else:
        logging.error(f"Cannot find sequence source for index {index}: neither {protein_seq_file} nor {protein_pdb_file} exists.")
        return False, None

    if not protein_sequence or any(x in protein_sequence for x in ["NOTFOUND", "ERROR", "NOTPARSED"]):
        logging.error(f"Invalid protein sequence for index {index}. Obtained: '{protein_sequence}'")
        return False, None

    # Retrieve ligand identifier: use CHAI1_LIGAND_IDENTIFIER if set, otherwise fallback to LIGAND
    ligand_id = getattr(self, 'CHAI1_LIGAND_IDENTIFIER', None)
    if not ligand_id:
        ligand_id = getattr(self, 'LIGAND', None)
    if not ligand_id or not isinstance(ligand_id, str):
        logging.error(f"Index {index}: Neither CHAI1_LIGAND_IDENTIFIER nor LIGAND is a valid string.")
        return False, None

    # Write FASTA with two entries: protein and ligand
    try:
        with open(fasta_path, 'w') as f:
            f.write(">protein|name=ChainA\n")
            f.write(f"{protein_sequence}\n")
            f.write(">ligand|name=ChainX\n")
            f.write(f"{ligand_id}\n")
        logging.info(f"Successfully generated Chai-1 FASTA at {fasta_path}")
        return True, len(protein_sequence)
    except IOError as e:
        logging.error(f"Failed to write FASTA file {fasta_path}: {e}")
        return False, None


def prepare_Chai1(self, index, cmd, gpu_id=None):
    """
    Prepares the shell command for running Chai-1 prediction.
    Sets up the working directory, generates the FASTA input, constructs the chai-lab command with options,
    and appends post-processing steps for converting the output CIF to PDB with added REMARKs.
    """
    logging.info(f"Preparing Chai-1 for index {index}.")

    # Check required attributes
    required_attrs = ['FOLDER_DESIGN', 'WT', 'all_scores_df', 'LIGAND']
    missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr) or getattr(self, attr) is None]
    if missing_attrs:
        logging.error(f"Missing required attributes for Chai-1 index {index}: {', '.join(missing_attrs)}")
        return cmd + f"\n# ERROR: Missing attributes for Chai-1: {', '.join(missing_attrs)}\n"

    try:
        index_str = str(index)
        folder_design = self.FOLDER_DESIGN
        wt_name = self.WT
        
        # Define working and output directories
        chai1_work_dir_py = os.path.join(folder_design, index_str, "Chai1")
        aizymes_index_dir_py = os.path.join(folder_design, index_str)
        os.makedirs(chai1_work_dir_py, exist_ok=True)

        # Define separate output directory inside working directory
        chai1_output_dir_py = os.path.join(chai1_work_dir_py, "output")
        if os.path.exists(chai1_output_dir_py):
            shutil.rmtree(chai1_output_dir_py)
        os.makedirs(chai1_output_dir_py)
        # Path for input FASTA
        fasta_input_path_py = os.path.join(chai1_work_dir_py, f"{wt_name}_Chai1_{index_str}.fasta")

        # Shell-quoting for safety
        chai1_work_dir_sh = shlex.quote(chai1_work_dir_py)
        fasta_input_path_sh = shlex.quote(fasta_input_path_py)
        chai1_output_dir_sh = shlex.quote(chai1_output_dir_py)
    except Exception as e:
        logging.error(f"Error defining paths for index {index}: {e}")
        return cmd + f"\n# ERROR: Path definition failed for Chai-1 index {index}: {e}\n"

    # Generate FASTA input
    fasta_success, protein_seq_len = _generate_chai1_fasta(self, index, fasta_input_path_py)
    if not fasta_success:
        logging.error(f"Failed to generate FASTA for Chai-1 index {index}.")
        return cmd + f"\n# ERROR: Failed to generate Chai-1 FASTA for index {index}.\n"

    try:
        # Calculate ligand residue number for later renumbering (protein length + 1)
        ligand_resi_py = (protein_seq_len + 1) if (protein_seq_len and isinstance(protein_seq_len, int)) else 9999
        # Determine ligand name from config (prefer CHAI1_LIGAND_IDENTIFIER, fallback to LIGAND, default 'LIG')
        ligand_name_py = getattr(self, 'LIGAND', 'LIG')
        ligand_name_py = str(ligand_name_py) if ligand_name_py else 'LIG'

        # Initialize command accumulator for this Chai-1 job
        current_cmd = f"\n### Chai-1 Prediction ###\n"

        # Activate specific environment if defined
        chai1_env_activate = getattr(self, 'CHAI1_ENV_ACTIVATE', None)
        if chai1_env_activate:
            current_cmd += f"# Activating Chai-1 environment\n{chai1_env_activate}\n"

        # Set GPU if provided
        if gpu_id is not None:
            current_cmd += f"export CUDA_VISIBLE_DEVICES={gpu_id}\n"
            logging.info(f"Chai-1 index {index} assigned to GPU: {gpu_id}")
        else:
            logging.info(f"Chai-1 index {index} running with default GPU/CPU settings.")

        # Build command parts
        chai1_cmd_parts = [
            "chai-lab", "fold", fasta_input_path_sh, chai1_output_dir_sh
        ]

        # Optional parameters
        num_samples = getattr(self, 'CHAI1_DIFFUSION_SAMPLES', 1)
        chai1_cmd_parts.extend(["--num-diffn-samples", str(num_samples)])

        num_recycles = getattr(self, 'CHAI1_TRUNK_RECYCLES', 3)
        chai1_cmd_parts.extend(["--num-trunk-recycles", str(num_recycles)])

        seed = getattr(self, 'SEED', None)
        if seed is not None:
            chai1_cmd_parts.extend(["--seed", str(seed + int(index))])

        if getattr(self, 'CHAI1_USE_MSA_SERVER', False):
            chai1_cmd_parts.append("--use-msa-server")
            msa_server_url = getattr(self, 'CHAI1_MSA_SERVER_URL', None)
            if msa_server_url:
                chai1_cmd_parts.extend(["--msa-server-url", shlex.quote(msa_server_url)])

        if getattr(self, 'CHAI1_USE_TEMPLATES_SERVER', False):
            chai1_cmd_parts.append("--use-templates-server")

        if getattr(self, 'CHAI1_LOW_MEMORY', True):
            chai1_cmd_parts.append("--low-memory")

        if getattr(self, 'CST_NAME', None) is not None:
            cst_file_py = os.path.join(self.FOLDER_PARENT, f"{self.CST_NAME}.csv")
            if os.path.exists(cst_file_py):
                chai1_cmd_parts.extend(["--constraint-path", shlex.quote(cst_file_py)])
                logging.info(f"Using constraints file for Chai-1: {cst_file_py}")
            else:
                logging.warning(f"Constraint file {cst_file_py} specified but not found for Chai-1.")

        current_cmd += shlex.join(chai1_cmd_parts) + "\n\n"

        # AWK scripts for ligand processing (primary: chain B resi 1, fallback: any LIG)
        # Using raw strings to avoid escaping issues
        awk_script_primary_py = r"""
BEGIN { if (new_resi+0 != new_resi) { new_resi=9999 } }
( ($1=="ATOM"||$1=="HETATM") && substr($0,18,3)=="LIG" && substr($0,22,1)=="B" && substr($0,23,4)+0==1 ) {
    serial   = substr($0,7,5); atom_name = substr($0,13,4); alt_loc   = substr($0,17,1);
    icode    = substr($0,27,1); x         = substr($0,31,8); y         = substr($0,39,8);
    z        = substr($0,47,8); occ       = substr($0,55,6); temp      = substr($0,61,6);
    element  = substr($0,77,2); charge    = substr($0,79,2);
    sub(/_.*/, "", atom_name);
    printf "ATOM  %5s %-4s%1s%3s %1s%4d%1s   %8s%8s%8s%6s%6s          %2s%2s\\n", \
           serial, atom_name, alt_loc, new_resn, new_chain, new_resi, icode, x, y, z, occ, temp, element, charge;
    found = 1
}
END { exit !found }
"""
        awk_script_fallback_py = r"""
BEGIN { if (new_resi+0 != new_resi) { new_resi=9999 } }
( ($1=="ATOM"||$1=="HETATM") && substr($0,18,3)=="LIG" ) {
    serial   = substr($0,7,5); atom_name = substr($0,13,4); alt_loc   = substr($0,17,1);
    icode    = substr($0,27,1); x         = substr($0,31,8); y         = substr($0,39,8);
    z        = substr($0,47,8); occ       = substr($0,55,6); temp      = substr($0,61,6);
    element  = substr($0,77,2); charge    = substr($0,79,2);
    sub(/_.*/, "", atom_name);
    printf "ATOM  %5s %-4s%1s%3s %1s%4d%1s   %8s%8s%8s%6s%6s          %2s%2s\\n", \
           serial, atom_name, alt_loc, new_resn, new_chain, new_resi, icode, x, y, z, occ, temp, element, charge;
    found = 1
}
END { exit !found }
"""

        # Generate REMARKs
        remarks_py = ""
        try:
            remarks_py = generate_remark_from_all_scores_df(self, index=index)
            logging.info(f"Generated REMARKs for Chai-1 index {index}")
        except Exception as e:
            logging.error(f"Error generating remarks for Chai-1 index {index}: {e}")
            remarks_py = "REMARK 999 Error generating remarks\n"

        # Define expected output files for post-processing
        model_idx = 0
        expected_output_cif_py = os.path.join(chai1_output_dir_py, f"pred.model_idx_{model_idx}.cif")
        expected_output_cif_sh = shlex.quote(expected_output_cif_py)
        expected_scores_npz_py = os.path.join(chai1_output_dir_py, f"scores.model_idx_{model_idx}.npz")
        expected_scores_npz_sh = shlex.quote(expected_scores_npz_py)

        final_structure_aizymes_name_py = os.path.join(aizymes_index_dir_py, f"{wt_name}_Chai1_{index_str}.pdb")
        final_structure_aizymes_name_sh = shlex.quote(final_structure_aizymes_name_py)
        final_scores_aizymes_name_py = os.path.join(aizymes_index_dir_py, f"scores_Chai1_{index_str}.npz")
        final_scores_aizymes_name_sh = shlex.quote(final_scores_aizymes_name_py)

        # Post-processing command
        post_process_cmd = f"""
# Post-process Chai-1 output
echo "Checking for Chai-1 output CIF: {expected_output_cif_sh}"
if [ -f {expected_output_cif_sh} ] && [ -s {expected_output_cif_sh} ]; then
  echo "Chai-1 output found. Converting CIF to PDB and adding remarks..."

  TMP_PDB_CONVERTED=$(mktemp)
  TMP_PDB_FINAL=$(mktemp)

  # 1. Convert CIF to PDB using the helper script
  echo "Running cif_to_pdb.py..."
  python "{self.FOLDER_PARENT}/cif_to_pdb.py" --cif_file {expected_output_cif_sh} --pdb_file "$TMP_PDB_CONVERTED"
  if [ $? -ne 0 ] || [ ! -s "$TMP_PDB_CONVERTED" ]; then
      echo "ERROR: cif_to_pdb.py conversion failed or produced empty file for {expected_output_cif_sh}" >&2
      rm -f "$TMP_PDB_CONVERTED" "$TMP_PDB_FINAL"
      exit 1
  fi
  echo "CIF converted to $TMP_PDB_CONVERTED"
  # Remove any TER records and hydrogens to clean the converted PDB
  sed -i '/TER/d' "$TMP_PDB_CONVERTED"
  sed -i '/ H  /d' "$TMP_PDB_CONVERTED"

  #############################
  # Standardise ligand records #
  #############################
  # Variables passed in from Python for ligand renaming/renumbering (already shell-safe)
  LIGAND_NAME=$(printf '%s' {shlex.quote(ligand_name_py)})
  LIGAND_RESI=$(printf '%s' {shlex.quote(str(ligand_resi_py))})
  REMARKS=$(printf '%s' {shlex.quote(remarks_py)})

  # 1. Start final PDB file with remarks header (match Boltz/ESMFold format)
  echo "REMARK   0 File processed from Chai-1 prediction" > "$TMP_PDB_FINAL"
  echo "REMARK   0 Chain A: Protein, Chain X: Ligand ($LIGAND_NAME)" >> "$TMP_PDB_FINAL"
  echo "$REMARKS" >> "$TMP_PDB_FINAL"

  # 2. Append protein ATOM lines (excluding original ligand) from converted PDB
  grep '^ATOM  ' "$TMP_PDB_CONVERTED" | grep -v " LIG " >> "$TMP_PDB_FINAL" || echo "Warning: No protein ATOM lines found or grep failed."
  echo "TER" >> "$TMP_PDB_FINAL"

  # 3. Process ligand lines via two-phase AWK (primary then fallback)
  LIGAND_LINES=$(mktemp)
  awk -v new_resn="$LIGAND_NAME" -v new_chain="X" -v new_resi="$LIGAND_RESI" \
      '{awk_script_primary_py}' \
      "$TMP_PDB_CONVERTED" > "$LIGAND_LINES"
  if [ $? -eq 0 ] && [ -s "$LIGAND_LINES" ]; then
      cat "$LIGAND_LINES" >> "$TMP_PDB_FINAL"
  else
      echo "Warning: primary ligand AWK failed, trying fallback..." >&2
      awk -v new_resn="$LIGAND_NAME" -v new_chain="X" -v new_resi="$LIGAND_RESI" \
          '{awk_script_fallback_py}' \
          "$TMP_PDB_CONVERTED" >> "$TMP_PDB_FINAL"
  fi
  rm -f "$LIGAND_LINES"
  echo "TER" >> "$TMP_PDB_FINAL"

  # 4. Finish file
  echo "END" >> "$TMP_PDB_FINAL"

  # 5. Move final PDB
  if [ -s "$TMP_PDB_FINAL" ]; then
      echo "Moving final PDB to {final_structure_aizymes_name_sh}"
      mv "$TMP_PDB_FINAL" {final_structure_aizymes_name_sh}
  else
      echo "ERROR: Final PDB assembly failed (empty file)." >&2
      rm -f "$TMP_PDB_CONVERTED" "$TMP_PDB_FINAL"
      exit 1
  fi
  rm -f "$TMP_PDB_CONVERTED"

  # 6. Copy score file
  if [ -f {expected_scores_npz_sh} ]; then
      echo "Copying Chai-1 scores NPZ to {final_scores_aizymes_name_sh}"
      cp {expected_scores_npz_sh} {final_scores_aizymes_name_sh}
  else
      echo "Warning: Chai-1 scores file not found: {expected_scores_npz_sh}" >&2
  fi
else
  echo "ERROR: Expected Chai-1 output CIF not found or empty: {expected_output_cif_sh}" >&2
  echo "Contents of Chai-1 output directory ({chai1_work_dir_sh}):"
  ls -la {chai1_work_dir_sh} || echo "Cannot access directory"
  exit 1
fi
"""
        
        current_cmd += post_process_cmd

        # Optional cleanup
        if getattr(self, 'REMOVE_TMP_FILES', False):
             current_cmd += f"\n# Removing temporary Chai-1 directory\necho \"Removing temporary directory: {chai1_work_dir_sh}\"\nrm -r {chai1_work_dir_sh}\n"

        logging.info(f"Chai-1 command for index {index} prepared successfully.")
        return cmd + current_cmd.strip() + "\n"
    except Exception as e:
        logging.error(f"Error preparing Chai-1 for index {index}: {e}")
        return cmd + f"\n# ERROR: Failed to prepare Chai-1 for index {index}: {e}\n" 