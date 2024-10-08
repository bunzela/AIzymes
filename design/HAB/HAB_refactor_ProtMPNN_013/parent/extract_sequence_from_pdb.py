
import argparse
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
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--pdb_in", type=str, help="PDB file from which the sequence is read.")
    argparser.add_argument("--sequence_out", type=str, help="File into which the sequence is storred.")

    args = argparser.parse_args()
    main(args)
    
