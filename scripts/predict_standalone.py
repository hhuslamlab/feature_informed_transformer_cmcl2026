"""
predict_standalone - Generate predictions from arbitrary source file
without requiring gold outputs or test data structure
"""
import argparse
import torch
from tqdm import tqdm

from decoding import get_decode_fn
from train import Trainer, Data, Arch, Decode
import util


def read_source_file(filepath):
    """Read source sequences from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def encode_source(source_lines, data):
    """Encode source sequences using the data vocabulary."""
    encoded = []
    for line in source_lines:
        # Split into characters or tokens based on data format
        tokens = line.split() if ' ' in line else list(line)
        # Encode using source vocabulary
        encoded_seq = [data.source_c2i.get(token, data.source_c2i.get('<unk>', 0)) 
                       for token in tokens]
        encoded.append(encoded_seq)
    return encoded


def create_batch(sequences, device):
    """Create a batch from a list of sequences."""
    # Pad sequences to same length
    max_len = max(len(seq) for seq in sequences)
    batch = []
    masks = []
    
    for seq in sequences:
        # Pad with 0 (typically PAD_IDX)
        padded = seq + [0] * (max_len - len(seq))
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        batch.append(padded)
        masks.append(mask)
    
    # Convert to tensors [seq_len, batch_size]
    batch_tensor = torch.tensor(batch, device=device).t()
    mask_tensor = torch.tensor(masks, device=device).t()
    
    return batch_tensor, mask_tensor


def main():
    """
    Generate predictions from an arbitrary source file
    """
    parser = argparse.ArgumentParser(description='Generate predictions from source file')
    parser.add_argument('--source', required=True, type=str, 
                       help='Source file to generate predictions for')
    parser.add_argument('--output', required=True, type=str,
                       help='Output file for predictions')
    parser.add_argument('--model_file', required=True, type=str,
                       help='Path to trained model file')
    parser.add_argument('--train', required=True, type=str, nargs='+',
                       help='Training data (needed for vocabulary)')
    parser.add_argument('--dev', required=True, type=str, nargs='+',
                       help='Dev data (needed for vocabulary)')
    parser.add_argument('--dataset', required=True, type=Data, choices=list(Data),
                       help='Dataset type')
    parser.add_argument('--arch', required=True, type=Arch, choices=list(Arch),
                       help='Model architecture')
    parser.add_argument('--bs', default=20, type=int,
                       help='Batch size for prediction')
    parser.add_argument('--max_decode_len', default=128, type=int,
                       help='Maximum decode length')
    parser.add_argument('--decode', default=Decode.greedy, type=Decode, 
                       choices=list(Decode), help='Decode method')
    parser.add_argument('--decode_beam_size', default=5, type=int,
                       help='Beam size for beam search')
    parser.add_argument('--indtag', default=False, action='store_true',
                       help='Separate tag from source string')
    
    args = parser.parse_args()
    
    # Create a minimal trainer just to load vocabularies and model
    trainer = Trainer()
    # Override some params
    trainer.params.dataset = args.dataset
    trainer.params.arch = args.arch
    trainer.params.indtag = args.indtag
    trainer.params.shuffle = False
    
    # Load data to build vocabularies (using empty test)
    empty_test = [args.source]  # Use source file as test
    trainer.load_data(args.dataset, args.train, args.dev, empty_test)
    
    # Load the trained model
    trainer.logger.info(f"Loading model from {args.model_file}")
    trainer.model = torch.load(args.model_file, map_location=trainer.device, 
                               weights_only=False)
    trainer.model = trainer.model.to(trainer.device)
    trainer.model.eval()
    
    # Get decode function
    decode_fn = get_decode_fn(args.decode, args.max_decode_len, args.decode_beam_size)
    
    # Read source file
    trainer.logger.info(f"Reading source from {args.source}")
    source_lines = read_source_file(args.source)
    trainer.logger.info(f"Found {len(source_lines)} source sequences")
    
    # Generate predictions
    predictions = []
    
    with open(args.output, "w", encoding='utf-8') as fp:
        with torch.no_grad():
            # Process in batches
            for i in tqdm(range(0, len(source_lines), args.bs)):
                batch_sources = source_lines[i:i + args.bs]
                
                # Encode sources using the data encoder
                # This is simpler - let the data loader handle encoding
                encoded_sources = []
                for src_line in batch_sources:
                    tokens = src_line.split() if ' ' in src_line else list(src_line)
                    encoded = trainer.data.encode_source(tokens)
                    encoded_sources.append(encoded)
                
                # Create batch
                src_batch, src_mask = create_batch(encoded_sources, trainer.device)
                
                # Generate predictions
                pred, _ = decode_fn(trainer.model, src_batch, src_mask)
                
                # Unpack and decode predictions
                pred = util.unpack_batch(pred)
                
                for src_line, p in zip(batch_sources, pred):
                    pred_str = trainer.data.decode_target(p)
                    fp.write(f'{src_line}\t{" ".join(pred_str)}\n')
                    predictions.append(pred_str)
    
    trainer.logger.info(f"Generated {len(predictions)} predictions")
    trainer.logger.info(f"Output written to {args.output}")


if __name__ == "__main__":
    main()


