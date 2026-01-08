"""
predict - Generate predictions from input without gold output comparison
"""
import torch
from tqdm import tqdm

from decoding import get_decode_fn
from train import Trainer
import util


def read_source_file(filepath):
    """Read source sequences from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    """
    Generate predictions from source input without gold output comparison
    """
    trainer = Trainer()
    params = trainer.params
    
    # Get decode function
    decode_fn = get_decode_fn(
        params.decode, params.max_decode_len, params.decode_beam_size
    )
    
    # Load data - this sets up vocabularies and data loaders
    # Note: We still need train/dev to build vocabularies, but test can be source-only
    trainer.load_data(params.dataset, params.train, params.dev, params.test)
    
    # Load the trained model
    assert params.load, "Must specify --load with path to trained model"
    trainer.load_model(params.load)
    trainer.model.eval()
    
    trainer.logger.info("Generating predictions...")
    
    # Generate predictions
    predictions = []
    
    # Iterate through test data
    sampler, nb_batch = trainer.iterate_batch('test', params.bs)
    
    with open(f"{params.model}.predictions.txt", "w", encoding='utf-8') as fp:
        with torch.no_grad():
            for src, src_mask, trg, trg_mask in tqdm(sampler(params.bs), total=nb_batch):
                # Generate predictions
                pred, _ = decode_fn(trainer.model, src, src_mask)
                
                # Unpack batch
                pred = util.unpack_batch(pred)
                src_list = util.unpack_batch(src)
                
                # Decode and write predictions
                for s, p in zip(src_list, pred):
                    src_str = trainer.data.decode_source(s)
                    pred_str = trainer.data.decode_target(p)
                    fp.write(f'{" ".join(src_str)}\t{" ".join(pred_str)}\n')
                    predictions.append(pred_str)
    
    trainer.logger.info(f"Generated {len(predictions)} predictions")
    trainer.logger.info(f"Output written to {params.model}.predictions.txt")


if __name__ == "__main__":
    main()


