"""
generate - Generate predictions from input file using only a checkpoint
No train/dev files needed - vocabularies are loaded from the model
"""
import argparse
import os
import sys
import torch
from tqdm import tqdm

# Ensure we import from the correct location (example/transformer/src)
# Must be at the FRONT of sys.path to override any other imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, script_dir)

# Remove any cached imports that might be wrong
for module in ['decoding', 'dataloader', 'util', 'transformer', 'model']:
    if module in sys.modules:
        del sys.modules[module]

from decoding import Decode, get_decode_fn
from dataloader import BOS, EOS, BOS_IDX, EOS_IDX, UNK_IDX
import util
import transformer


def read_source_file(filepath, split_on_hash=False):
    """Read source sequences from a file.
    
    Args:
        filepath: Path to input file
        split_on_hash: If True, split lines on '#' to get multiple examples per line
    
    Format examples:
    - Single example per line: "l e m m a <TAG1;TAG2>"
    - Multiple examples per line: "lemma1 <tags1> # lemma2 <tags2> # <tags3>"
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        examples = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split on '#' if requested (training file format)
            if split_on_hash and '#' in line:
                parts = [p.strip() for p in line.split('#') if p.strip()]
                examples.extend(parts)
            else:
                examples.append(line)
        
        return examples


def parse_input_line(line):
    """Parse input line to extract lemma and tags.
    Format: 'l e m m a <TAG1;TAG2;TAG3>' or '<TAG1;TAG2;TAG3>' (for query only)
    
    Examples from training file:
    - "t ɾ a d ˈ u s e n <V;IND;PRS;3;PL>"  -> lemma + tags
    - "<V;SBJV;PRS;3;PL>"  -> just tags (query target form)
    """
    import re
    
    line = line.strip()
    
    # Match pattern: optional lemma followed by tags in angle brackets
    # Pattern: (characters)? <tags>
    match = re.match(r'^(.*?)\s*<(.+?)>\s*$', line)
    
    if match:
        lemma_str = match.group(1).strip()
        tags_str = match.group(2).strip()
        tags = [t.strip() for t in tags_str.split(';') if t.strip()]
        
        # Parse lemma if present
        if lemma_str:
            lemma_tokens = lemma_str.split()
        else:
            # No lemma provided, just tags (target query only)
            lemma_tokens = []
    else:
        # No angle brackets - assume it's just lemma without tags
        lemma_tokens = line.split() if ' ' in line else list(line)
        tags = []
    
    return lemma_tokens, tags


def encode_sources(source_lines, src_c2i, attr_c2i, debug=False):
    """Encode source sequences using vocabulary from model.
    Uses the same encoding as dataloader.encode_source (lines 139-151)
    """
    encoded = []
    for idx, line in enumerate(source_lines):
        lemma_tokens, tags = parse_input_line(line)
        
        # Build the source sequence as tokens: [BOS, tag1, tag2, ..., char1, char2, ..., EOS]
        sent = [BOS]
        if tags:
            sent.extend(tags)
        sent.extend(lemma_tokens)
        sent.append(EOS)
        
        # Now encode using the same logic as dataloader.encode_source (lines 146-150)
        # Check source_c2i first, then attr_c2i
        s = []
        for x in sent:
            if x in src_c2i:
                s.append(src_c2i[x])
            elif attr_c2i and x in attr_c2i:
                s.append(attr_c2i[x])
            else:
                # This shouldn't happen for BOS/EOS, but handle unknown tokens
                if x == BOS:
                    s.append(BOS_IDX)
                elif x == EOS:
                    s.append(EOS_IDX)
                else:
                    s.append(UNK_IDX)
        
        # Debug output for first example
        if debug and idx == 0:
            print(f"\nDebug: Encoded sequence for first example:")
            print(f"  Input: {line}")
            print(f"  Parsed lemma: {lemma_tokens}")
            print(f"  Parsed tags: {tags}")
            print(f"  Token sequence: {sent}")
            print(f"  Encoded indices: {s}")
            print(f"  Length: {len(s)}")
        
        encoded.append(s)
    return encoded


def create_batch(sequences, device):
    """Create a batch tensor from list of sequences."""
    if not sequences:
        return None, None
    
    # Pad sequences to same length
    max_len = max(len(seq) for seq in sequences)
    batch = []
    masks = []
    
    for seq in sequences:
        # Pad with 0 (PAD_IDX)
        padded = seq + [0] * (max_len - len(seq))
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        batch.append(padded)
        masks.append(mask)
    
    # Convert to tensors [seq_len, batch_size]
    batch_tensor = torch.tensor(batch, device=device).t()
    mask_tensor = torch.tensor(masks, device=device).t()
    
    return batch_tensor, mask_tensor


def decode_target(pred_indices, target_vocab):
    """Decode prediction indices to characters using target vocabulary.
    Same as dataloader.decode_target but for list input.
    """
    return [target_vocab[x] for x in pred_indices]


def main():
    """
    Generate predictions from an input file using only a model checkpoint
    """
    # Verify we're using the correct decoding module
    import decoding
    if 'decode_greedy_transformer' in dir(decoding):
        import inspect
        src_file = inspect.getfile(decoding)
        if 'example/transformer/src' not in src_file and 'example\\transformer\\src' not in src_file:
            print(f"WARNING: Using wrong decoding.py from: {src_file}")
            print("Please run this script from: neural-transducer/example/transformer/")
            print("  cd neural-transducer/example/transformer")
            print("  python src/generate.py ...")
            sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description='Generate predictions using only model checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Input file format:
  Single example per line:
    "l e m m a <V;IND;PRS;3;SG>"
  
  Multiple examples per line (use --split_on_hash):
    "lemma1 <tags1> # lemma2 <tags2> # <tags3>"
  
  Target query only (no lemma):
    "<V;IND;PRS;3;SG>"

IMPORTANT: Run this script from the example/transformer/ directory:
  cd neural-transducer/example/transformer
  python src/generate.py --source file.src --output out.txt --checkpoint model.pt ...
'''
    )
    parser.add_argument('--source', required=True, type=str,
                       help='Source file with one input per line')
    parser.add_argument('--output', required=True, type=str,
                       help='Output file for predictions (one per line)')
    parser.add_argument('--checkpoint', required=True, type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', default=32, type=int,
                       help='Batch size for prediction')
    parser.add_argument('--max_len', default=128, type=int,
                       help='Maximum decode length')
    parser.add_argument('--decode_method', default='greedy', 
                       choices=['greedy', 'beam'],
                       help='Decode method')
    parser.add_argument('--beam_size', default=5, type=int,
                       help='Beam size for beam search')
    parser.add_argument('--split_on_hash', action='store_true',
                       help='Split input lines on # to extract multiple examples (training file format)')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug information about input parsing')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print(f"Loading model from {args.checkpoint}")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    
    # Extract vocabularies from model
    if not hasattr(model, 'src_c2i') or not hasattr(model, 'trg_c2i'):
        raise ValueError("Model checkpoint does not contain vocabularies (src_c2i, trg_c2i). "
                        "Cannot generate predictions without vocabularies.")
    
    src_c2i = model.src_c2i
    trg_c2i = model.trg_c2i
    attr_c2i = model.attr_c2i if hasattr(model, 'attr_c2i') else None
    
    # Create inverse mapping for target vocabulary (list indexed by integer)
    target_vocab = [''] * len(trg_c2i)
    for char, idx in trg_c2i.items():
        target_vocab[idx] = char
    
    print(f"Source vocabulary size: {len(src_c2i)}")
    print(f"Target vocabulary size: {len(trg_c2i)}")
    if attr_c2i:
        print(f"Attribute vocabulary size: {len(attr_c2i)}")
    else:
        print("No attribute vocabulary found")
    
    # Setup decoder
    decode_type = Decode.greedy if args.decode_method == 'greedy' else Decode.beam
    decode_fn = get_decode_fn(decode_type, args.max_len, args.beam_size)
    
    # Read source file
    print(f"Reading source from {args.source}")
    source_lines = read_source_file(args.source, split_on_hash=args.split_on_hash)
    print(f"Found {len(source_lines)} source sequences")
    
    if args.debug and len(source_lines) > 0:
        print("\nDebug: First 3 input examples:")
        for i, line in enumerate(source_lines[:3]):
            lemma, tags = parse_input_line(line)
            print(f"  {i+1}. Input: {line}")
            print(f"     Parsed lemma: {lemma}")
            print(f"     Parsed tags: {tags}")
    
    if not source_lines:
        print("No source lines found!")
        return
    
    # Generate predictions
    all_predictions = []
    
    print("Generating predictions...")
    with open(args.output, "w", encoding='utf-8') as fp:
        with torch.no_grad():
            # Process in batches
            for i in tqdm(range(0, len(source_lines), args.batch_size)):
                batch_sources = source_lines[i:i + args.batch_size]
                
                # Encode sources
                encoded_sources = encode_sources(batch_sources, src_c2i, attr_c2i, 
                                                debug=(args.debug and i == 0))
                
                # Create batch tensors
                src_batch, src_mask = create_batch(encoded_sources, device)
                
                # Generate predictions (same as test.py)
                pred, _ = decode_fn(model, src_batch, src_mask)
                
                # Unpack batch (same as test.py - handles both tensor and list)
                pred = util.unpack_batch(pred)
                
                # Decode predictions to characters
                for p in pred:
                    pred_chars = decode_target(p, target_vocab)
                    pred_str = ' '.join(pred_chars)
                    fp.write(f'{pred_str}\n')
                    all_predictions.append(pred_str)
    
    print(f"Generated {len(all_predictions)} predictions")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()

