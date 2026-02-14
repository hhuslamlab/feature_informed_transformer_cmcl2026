"""
test

Usage:
    python test.py --dataset <dataset> --train <train_files> --dev <dev_files> \
                   --test <test_files> --model <model_prefix> --load <checkpoint> \
                   [--eval_dev] [--eval_test]

Options:
    --eval_dev      Evaluate on dev set (default: True if neither flag specified)
    --eval_test     Evaluate on test set (default: True if neither flag specified)
"""
import sys
import os

# Ensure we import from this directory's modules, not from other locations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from decoding import get_decode_fn
from train import Trainer
from tqdm import tqdm
import util


class TestTrainer(Trainer):
    """Extended Trainer with separate dev/test evaluation flags."""

    def set_args(self):
        super().set_args()
        self.parser.add_argument(
            '--eval_dev',
            action='store_true',
            help='Evaluate on dev set'
        )
        self.parser.add_argument(
            '--eval_test',
            action='store_true',
            help='Evaluate on test set'
        )


    def reload_and_test_selective(self, model_fp, best_fp, batch_size, decode_fn,
                                   eval_dev=True, eval_test=True):
        """Reload model and test on selected datasets."""
        self.model = None
        self.logger.info(f"loading {best_fp} for testing")
        self.load_model(best_fp)
        dec_bs = min(32, batch_size)

        if eval_dev:
            self.logger.info("decoding dev set")
            results = self.decode("dev", dec_bs, f"{model_fp}.decode", decode_fn)
            torch.cuda.empty_cache()
            if results:
                for result in results:
                    self.logger.info(f"DEV {result.long_desc} is {result.res} at epoch -1")
                results_str = " ".join([f"{r.desc} {r.res}" for r in results])
                self.logger.info(f'DEV {model_fp.split("/")[-1]} {results_str}')

        if eval_test and self.data.test_file is not None:
            self.logger.info("decoding test set")
            results = self.decode("test", dec_bs, f"{model_fp}.decode", decode_fn)
            torch.cuda.empty_cache()
            if results:
                for result in results:
                    self.logger.info(f"TEST {result.long_desc} is {result.res} at epoch -1")
                results_str = " ".join([f"{r.desc} {r.res}" for r in results])
                self.logger.info(f'TEST {model_fp.split("/")[-1]} {results_str}')


def main():
    """
    main
    """
    trainer = TestTrainer()
    params = trainer.params
    decode_fn = get_decode_fn(
        params.decode, params.max_decode_len, params.decode_beam_size
    )
    trainer.load_data(params.dataset, params.train, params.dev, params.test)
    trainer.setup_evalutator()

    if not params.load:
        raise ValueError("--load is required: specify the model checkpoint to load (e.g., --load model.nll_0.5.epoch_50)")

    # If neither flag is specified, run both (default behavior)
    eval_dev = params.eval_dev
    eval_test = params.eval_test
    if not eval_dev and not eval_test:
        eval_dev = True
        eval_test = True

    trainer.reload_and_test_selective(
        params.model, params.load, params.bs, decode_fn,
        eval_dev=eval_dev, eval_test=eval_test
    )


if __name__ == "__main__":
    main()
