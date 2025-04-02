def make_main_scripts(self):

    # ------------------------------------------------------------------------------------------------------------------------------
    # Create the cif_to_pdb.py script
    # ------------------------------------------------------------------------------------------------------------------------------
    with open(f'{self.FOLDER_PARENT}/cif_to_pdb.py', 'w') as f:
        f.write("""
import gemmi
import argparse

def main(args):

    cif_file = args.cif_file
    pdb_file = args.pdb_file
    
    doc = gemmi.cif.read_file(cif_file)
    block = doc.sole_block()  
    structure = gemmi.make_structure_from_block(block)
    structure.setup_entities()  
    with open(pdb_file, "w") as f:
        f.write(structure.make_pdb_string())
    
if __name__ == "__main__":

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--cif_file", type=str, help="Input CIF file.")
    argparser.add_argument("--pdb_file", type=str, help="Output PDB file.")

    args = argparser.parse_args()
    main(args)
""")

    # ------------------------------------------------------------------------------------------------------------------------------
    # Create the ESMfold.py script
    # ------------------------------------------------------------------------------------------------------------------------------
    with open(f'{self.FOLDER_PARENT}/ESMfold.py', 'w') as f:
        f.write("""
import argparse
import sys
from transformers import AutoTokenizer, EsmForProteinFolding, EsmConfig
import torch
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

def main(args):

    sequence_file = args.sequence_file
    output_file = args.output_file

    # Set PyTorch to use only one thread
    torch.set_num_threads(1)

    with open(sequence_file) as f: sequence=f.read()

    def convert_outputs_to_pdb(outputs):
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            )
            pdbs.append(to_pdb(pred))
        return pdbs

    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        model.trunk.set_chunk_size(64)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        model = model.to(device)
        tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)
        
        with torch.no_grad(): 
            output = model(tokenized_input)

        pdb = convert_outputs_to_pdb(output)
        with open(output_file, "w") as f: 
            f.write("".join(pdb))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--sequence_file", type=str, help="File containing sequence to be predicted.")
    argparser.add_argument("--output_file", type=str, help="Output PDB.")

    args = argparser.parse_args()
    main(args)
""")


    # ------------------------------------------------------------------------------------------------------------------------------
    # Create the extract_sequence_from_pdb.py script
    # ------------------------------------------------------------------------------------------------------------------------------
    with open(f'{self.FOLDER_PARENT}/extract_sequence_from_pdb.py', 'w') as f:
        f.write("""
import argparse
import warnings
from Bio import BiopythonParserWarning
from Bio import SeqIO

def main(args):

    pdb_in = args.pdb_in
    sequence_out = args.sequence_out
         
    with open(pdb_in, "r") as f:
        for record in SeqIO.parse(f, "pdb-atom"):
            seq = str(record.seq)
    
    with open(sequence_out, "w") as f:
        f.write(seq)
 
if __name__ == "__main__":
    
    warnings.simplefilter("ignore", BiopythonParserWarning)
    
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--pdb_in", type=str, help="PDB file from which the sequence is read.")
    argparser.add_argument("--sequence_out", type=str, help="File into which the sequence is storred.")
     
    args = argparser.parse_args()
    main(args)
    
""")

    # ------------------------------------------------------------------------------------------------------------------------------
    # Create the find_highest_scoring_sequence.py script
    # ------------------------------------------------------------------------------------------------------------------------------
    with open(f'{self.FOLDER_PARENT}/find_highest_scoring_sequence.py', 'w') as f:
        f.write('''
import re
import argparse

def main(args):

    sequence_wildcard = args.sequence_wildcard
    sequence_parent   = args.sequence_parent
    sequence_in       = args.sequence_in
    sequence_out      = args.sequence_out
    
    # Read the parent sequence
    with open(sequence_parent, 'r') as file:
        sequence_parent = file.readline().strip()

    # Read the input sequence pattern and prepare it for regex matching
    with open(args.sequence_wildcard, 'r') as file:
        sequence_wildcard = file.readline().strip()
    sequence_wildcard = re.sub('X', '.', sequence_wildcard)  # Replace 'X' with regex wildcard '.'

    highest_score = 0
    highest_scoring_sequence = ''

    # Process the sequence file to find the highest scoring sequence
    with open(sequence_in, 'r') as file:
        for line in file:
            if line.startswith('>'):
                score_match = re.search('global_score=(\d+\.\d+)', line)
                if score_match:
                    score = float(score_match.group(1))
                    sequence = next(file, '').strip()  # Read the next line for the sequence
                    
                    # Check if the score is higher, the sequence is different from the parent,
                    # and does not match the input sequence pattern
                    if score > highest_score and sequence != sequence_parent and not re.match(sequence_wildcard, sequence):
                        highest_score = score
                        highest_scoring_sequence = sequence

    with open(sequence_out, 'w') as f:
        f.write(highest_scoring_sequence)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--sequence_wildcard", type=str, help="Sequence file with wildcards for designable residues.")
    argparser.add_argument("--sequence_parent", type=str, help="Sequence file of design parent variant.")
    argparser.add_argument("--sequence_in", type=str, help="Sequence file containing all designed variants.")
    argparser.add_argument("--sequence_out", type=str, help="Output sequence file of best variant.")

    args = argparser.parse_args()
    main(args)

''')       
        
    # ------------------------------------------------------------------------------------------------------------------------------
    # Create the process_boltz_results.py script
    # ------------------------------------------------------------------------------------------------------------------------------
    with open(f'{self.FOLDER_PARENT}/process_boltz_results.py', 'w') as f:
        f.write("""
import argparse
import os
import glob
import shutil
import json
from pathlib import Path
import sys
import subprocess

def main(args):
    \"\"\"Process Boltz prediction results and copy the final model to the expected location.\"\"\"
    
    boltz_dir = args.boltz_dir
    output_pdb = args.output_pdb
    
    # Find the most recent results directory (it starts with 'boltz_results_')
    results_dirs = sorted(glob.glob(os.path.join(boltz_dir, 'boltz_results_*')), key=os.path.getmtime, reverse=True)
    
    if not results_dirs:
        # Try using shell pattern matching as a fallback
        try:
            results_dirs_cmd = subprocess.run(
                f"ls -td {os.path.join(boltz_dir, 'boltz_results_*')} 2>/dev/null", 
                shell=True, capture_output=True, text=True
            )
            if results_dirs_cmd.returncode == 0 and results_dirs_cmd.stdout.strip():
                results_dirs = results_dirs_cmd.stdout.strip().split('\n')
                print(f"Found results directories using shell pattern: {results_dirs}")
            else:
                print(f"Error: No boltz_results_* directory found in {boltz_dir}", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to find boltz_results_* directory: {e}", file=sys.stderr)
            sys.exit(1)
        
    results_dir = results_dirs[0]
    print(f"Using results directory: {results_dir}")
    
    # Find the prediction directory
    prediction_dirs = glob.glob(os.path.join(results_dir, 'predictions', '*'))
    
    if not prediction_dirs:
        # Try using shell pattern matching as a fallback
        try:
            prediction_dirs_cmd = subprocess.run(
                f"ls -d {os.path.join(results_dir, 'predictions', '*')} 2>/dev/null", 
                shell=True, capture_output=True, text=True
            )
            if prediction_dirs_cmd.returncode == 0 and prediction_dirs_cmd.stdout.strip():
                prediction_dirs = prediction_dirs_cmd.stdout.strip().split('\n')
                print(f"Found prediction directories using shell pattern: {prediction_dirs}")
            else:
                print(f"Error: No prediction directory found in {results_dir}/predictions", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to find prediction directory: {e}", file=sys.stderr)
            sys.exit(1)
        
    prediction_dir = prediction_dirs[0]
    run_id = os.path.basename(prediction_dir)
    
    # Look for the model_0.pdb file
    model_path = os.path.join(prediction_dir, f"{run_id}_model_0.pdb")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found", file=sys.stderr)
        sys.exit(1)
        
    # Copy the model to the output location
    os.makedirs(os.path.dirname(output_pdb), exist_ok=True)
    shutil.copy(model_path, output_pdb)
    print(f"Successfully copied model from {model_path} to {output_pdb}")
    
    # Check for confidence scores
    confidence_path = os.path.join(prediction_dir, f"confidence_{run_id}_model_0.json")
    if os.path.exists(confidence_path):
        confidence_output = os.path.join(os.path.dirname(output_pdb), 
                                        f"{os.path.splitext(os.path.basename(output_pdb))[0]}_confidence.json")
        shutil.copy(confidence_path, confidence_output)
        print(f"Copied confidence scores to {confidence_output}")
        
        # Print summary of confidence scores for logging
        try:
            with open(confidence_path, 'r') as f:
                confidence = json.load(f)
                if isinstance(confidence, dict):
                    for key, value in confidence.items():
                        if isinstance(value, (int, float)):
                            print(f"Confidence {key}: {value}")
        except Exception as e:
            print(f"Warning: Failed to read confidence scores: {e}", file=sys.stderr)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Boltz results and copy the final model.")
    parser.add_argument("--boltz_dir", type=str, required=True, help="Directory containing Boltz results")
    parser.add_argument("--output_pdb", type=str, required=True, help="Path to save the final PDB model")
    
    args = parser.parse_args()
    main(args)
""")
        