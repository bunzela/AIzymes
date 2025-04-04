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
        