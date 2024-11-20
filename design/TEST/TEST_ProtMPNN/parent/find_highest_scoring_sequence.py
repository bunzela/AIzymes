
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

