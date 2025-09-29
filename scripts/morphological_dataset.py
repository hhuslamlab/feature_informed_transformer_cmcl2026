#!/usr/bin/env python3
"""
Optimized dataset class for morphological reinflection task
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import re
import numpy as np

class MorphologicalDataset(Dataset):
    """Optimized dataset for morphological reinflection task"""
    
    def __init__(self, src_file: str, tgt_file: str, src_vocab: Dict[str, int], 
                 tgt_vocab: Dict[str, int], max_length: int = 100):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        # Load and preprocess data for faster access
        self.data = self._load_and_preprocess_data()
        
    def _load_and_preprocess_data(self) -> List[Tuple[List[int], List[int], List[int], List[int]]]:
        """Load and preprocess data with tokenization for faster access"""
        data = []
        
        # Pre-compute special token indices
        pad_idx = self.src_vocab['<PAD>']
        unk_idx = self.src_vocab['<UNK>']
        bos_idx = self.src_vocab['<BOS>']
        eos_idx = self.src_vocab['<EOS>']
        
        with open(self.src_file, 'r', encoding='utf-8') as src_f, \
             open(self.tgt_file, 'r', encoding='utf-8') as tgt_f:
            
            for src_line, tgt_line in zip(src_f, tgt_f):
                src_tokens = src_line.strip().split()
                tgt_tokens = tgt_line.strip().split()
                
                # Filter out empty lines
                if src_tokens and tgt_tokens:
                    # Pre-tokenize sequences
                    src_indices, src_mask = self._tokenize_sequence_fast(
                        src_tokens, self.src_vocab, pad_idx, unk_idx, bos_idx, eos_idx
                    )
                    tgt_indices, tgt_mask = self._tokenize_sequence_fast(
                        tgt_tokens, self.tgt_vocab, pad_idx, unk_idx, bos_idx, eos_idx
                    )
                    
                    data.append((src_indices, src_mask, tgt_indices, tgt_mask))
        
        return data
    
    def _tokenize_sequence_fast(self, tokens: List[str], vocab: Dict[str, int], 
                               pad_idx: int, unk_idx: int, bos_idx: int, eos_idx: int) -> Tuple[List[int], List[int]]:
        """Fast tokenization with pre-computed indices"""
        # Convert tokens to indices using vectorized operations
        indices = []
        for token in tokens:
            indices.append(vocab.get(token, unk_idx))
        
        # Add BOS and EOS
        indices = [bos_idx] + indices + [eos_idx]
        
        # Truncate or pad to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [pad_idx] * (self.max_length - len(indices))
        
        # Create mask (1 for real tokens, 0 for padding)
        mask = [1 if idx != pad_idx else 0 for idx in indices]
        
        return indices, mask
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        return self.data[idx]

def build_vocabulary(data_files: List[str], min_freq: int = 1) -> Dict[str, int]:
    """Build vocabulary from data files with optimized processing"""
    token_freq = {}
    
    for file_path in data_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                for token in tokens:
                    token_freq[token] = token_freq.get(token, 0) + 1
    
    # Filter by frequency and sort
    filtered_tokens = [(token, freq) for token, freq in token_freq.items() 
                       if freq >= min_freq]
    filtered_tokens.sort(key=lambda x: x[1], reverse=True)
    
    # Build vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
    
    # Add tokens to vocabulary
    for token, _ in filtered_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    
    return vocab

def tokenize_sequence(tokens: List[str], vocab: Dict[str, int], 
                     max_length: int, add_bos_eos: bool = True) -> Tuple[List[int], List[int]]:
    """Tokenize a sequence and create masks"""
    # Convert tokens to indices
    indices = []
    for token in tokens:
        if token in vocab:
            indices.append(vocab[token])
        else:
            indices.append(vocab['<UNK>'])
    
    # Add BOS and EOS if requested
    if add_bos_eos:
        indices = [vocab['<BOS>']] + indices + [vocab['<EOS>']]
    
    # Truncate or pad to max_length
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices = indices + [vocab['<PAD>']] * (max_length - len(indices))
    
    # Create mask (1 for real tokens, 0 for padding)
    mask = [1 if idx != vocab['<PAD>'] else 0 for idx in indices]
    
    return indices, mask

def collate_fn(batch: List[Tuple[List[int], List[int], List[int], List[int]]], 
               src_vocab: Dict[str, int], tgt_vocab: Dict[str, int],
               max_length: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized collate function for DataLoader with vectorized operations"""
    
    # Pre-allocate lists for better memory management
    batch_size = len(batch)
    src_batch = []
    src_masks = []
    tgt_batch = []
    tgt_masks = []
    
    # Extract pre-tokenized data
    for src_indices, src_mask, tgt_indices, tgt_mask in batch:
        src_batch.append(src_indices)
        src_masks.append(src_mask)
        tgt_batch.append(tgt_indices)
        tgt_masks.append(tgt_mask)
    
    # Convert to tensors using stack for better performance
    src_batch = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in src_batch])
    src_masks = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in src_masks])
    tgt_batch = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in tgt_batch])
    tgt_masks = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in tgt_masks])
    
    # Transpose for transformer input format [seq_len, batch_size]
    src_batch = src_batch.t()
    src_masks = src_masks.t()
    tgt_batch = tgt_batch.t()
    tgt_masks = tgt_masks.t()
    
    return src_batch, src_masks, tgt_batch, tgt_masks

def analyze_vocabulary(data_files: List[str]) -> Dict:
    """Analyze vocabulary statistics"""
    token_freq = {}
    total_tokens = 0
    total_sequences = 0
    
    for file_path in data_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                total_sequences += 1
                for token in tokens:
                    token_freq[token] = token_freq.get(token, 0) + 1
                    total_tokens += 1
    
    # Analyze special tokens (features)
    feature_tokens = [token for token in token_freq.keys() 
                     if token.startswith('<') and token.endswith('>')]
    
    # Analyze character tokens
    char_tokens = [token for token in token_freq.keys() 
                  if not token.startswith('<') and len(token) == 1]
    
    return {
        'total_tokens': total_tokens,
        'total_sequences': total_sequences,
        'unique_tokens': len(token_freq),
        'feature_tokens': len(feature_tokens),
        'char_tokens': len(char_tokens),
        'avg_seq_length': total_tokens / total_sequences if total_sequences > 0 else 0,
        'feature_examples': feature_tokens[:10],  # First 10 features
        'char_examples': char_tokens[:10]  # First 10 characters
    }
