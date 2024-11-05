
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

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(64)
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
    with torch.no_grad(): output = model(tokenized_input)
    pdb = convert_outputs_to_pdb(output)
    with open(output_file, "w") as f: f.write("".join(pdb))
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--sequence_file", type=str, help="File containing sequence to be predicted.")
    argparser.add_argument("--output_file", type=str, help="Output PDB.")

    args = argparser.parse_args()
    main(args)
