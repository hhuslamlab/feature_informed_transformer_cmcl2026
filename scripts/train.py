import argparse
import json
import logging
import math
import os
import time
import gc
from pathlib import Path
from typing import Dict, Tuple, Optional

# CPU optimizations - MUST be set before importing torch
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# CPU optimizations
torch.set_num_threads(os.cpu_count())  # Use all CPU cores
torch.set_num_interop_threads(1)  # Single interop thread for better performance

from transformer import TagTransformer, PAD_IDX, DEVICE
from morphological_dataset import MorphologicalDataset, build_vocabulary, collate_fn

# Disable all logging for speed
logging.disable(logging.CRITICAL)

def create_cpu_optimized_model(config: Dict, src_vocab: Dict[str, int], tgt_vocab: Dict[str, int]) -> TagTransformer:
    """Create model with maximum CPU optimizations (compilation disabled for compatibility)"""
    
    feature_tokens = [token for token in src_vocab.keys() 
                     if token.startswith('<') and token.endswith('>')]
    nb_attr = len(feature_tokens)
    
    model = TagTransformer(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(tgt_vocab),
        embed_dim=config['embed_dim'],
        nb_heads=config['nb_heads'],
        src_hid_size=config['src_hid_size'],
        src_nb_layers=config['src_nb_layers'],
        trg_hid_size=config['trg_hid_size'],
        trg_nb_layers=config['trg_nb_layers'],
        dropout_p=config['dropout_p'],
        tie_trg_embed=config['tie_trg_embed'],
        label_smooth=config['label_smooth'],
        nb_attr=nb_attr,
        src_c2i=src_vocab,
        trg_c2i=tgt_vocab,
        attr_c2i={},
    )
    
    # Aggressive weight initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        elif p.dim() == 1:
            nn.init.uniform_(p, -0.1, 0.1)
    
    # Model compilation disabled for compatibility with older g++ versions
    # This avoids the "unrecognized command line option '-std=c++17'" error
    print("✓ Model created (compilation disabled for compatibility)")
    
    return model

def create_simple_dataloader(dataset, config: Dict, src_vocab: Dict, tgt_vocab: Dict, device: torch.device):
    """Create DataLoader with appropriate settings for CPU/GPU"""
    
    # Define collate function outside to avoid pickling issues
    def collate_wrapper(batch):
        return collate_fn(batch, src_vocab, tgt_vocab, config['max_length'])
    
    # Use device-appropriate settings
    if device.type == 'cuda':
        # Use 0 workers to avoid DataLoader freezing issues
        # pin_memory=True helps with GPU data transfer
        dataloader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            collate_fn=collate_wrapper,
            num_workers=0,  # Use 0 workers to avoid freezing on limited systems
            pin_memory=True,  # Enable for GPU data transfer optimization
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            collate_fn=collate_wrapper,
            num_workers=0,  # No multiprocessing to avoid issues on CPU
            pin_memory=False,  # Disable for CPU
            drop_last=True,
        )
    
    return dataloader

def train_epoch_cpu(model: TagTransformer, 
                    dataloader: DataLoader, 
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    config: Dict) -> float:
    """CPU-optimized training with minimal overhead"""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Use set_to_none for faster gradient clearing
    optimizer.zero_grad(set_to_none=True)
    
    start_time = time.time()
    
    for batch_idx, (src, src_mask, tgt, tgt_mask) in enumerate(dataloader):
        # Move to device with appropriate non_blocking setting
        non_blocking = device.type == 'cuda'
        src = src.to(device, non_blocking=non_blocking)
        src_mask = src_mask.to(device, non_blocking=non_blocking)
        tgt = tgt.to(device, non_blocking=non_blocking)
        tgt_mask = tgt_mask.to(device, non_blocking=non_blocking)
        
        # Forward pass
        output = model(src, src_mask, tgt, tgt_mask)
        loss = model.loss(output[:-1], tgt[1:])
        
        # Backward pass
        loss.backward()
        
        # Optimizer step every batch (no accumulation for speed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Minimal logging - only every 100 batches
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config['batch_size'] / elapsed
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Speed: {samples_per_sec:.0f} samples/sec')
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate_cpu(model: TagTransformer, 
                 dataloader: DataLoader, 
                 device: torch.device,
                 config: Dict) -> float:
    """CPU-optimized validation"""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for src, src_mask, tgt, tgt_mask in dataloader:
            non_blocking = device.type == 'cuda'
            src = src.to(device, non_blocking=non_blocking)
            src_mask = src_mask.to(device, non_blocking=non_blocking)
            tgt = tgt.to(device, non_blocking=non_blocking)
            tgt_mask = tgt_mask.to(device, non_blocking=non_blocking)
            
            output = model(src, src_mask, tgt, tgt_mask)
            loss = model.loss(output[:-1], tgt[1:])
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_test_cpu(model: TagTransformer, 
                     dataloader: DataLoader, 
                     device: torch.device,
                     config: Dict) -> float:
    """CPU-optimized test evaluation"""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    print("Evaluating on test set...")
    
    with torch.no_grad():
        for batch_idx, (src, src_mask, tgt, tgt_mask) in enumerate(dataloader):
            non_blocking = device.type == 'cuda'
            src = src.to(device, non_blocking=non_blocking)
            src_mask = src_mask.to(device, non_blocking=non_blocking)
            tgt = tgt.to(device, non_blocking=non_blocking)
            tgt_mask = tgt_mask.to(device, non_blocking=non_blocking)
            
            output = model(src, src_mask, tgt, tgt_mask)
            loss = model.loss(output[:-1], tgt[1:])
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress indicator for test evaluation
            if batch_idx % 50 == 0:
                print(f"  Test batch {batch_idx}/{len(dataloader)}")
    
    avg_loss = total_loss / num_batches
    return avg_loss

def save_checkpoint_cpu(model: TagTransformer, 
                       optimizer: optim.Optimizer, 
                       epoch: int, 
                       loss: float, 
                       save_path: str,
                       updates: int = None,
                       data_paths: dict = None):
    """Fast checkpoint saving with data paths"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if updates is not None:
        checkpoint['updates'] = updates
    
    if data_paths is not None:
        checkpoint['data_paths'] = data_paths
    
    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved: {save_path}')

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the output directory"""
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        return None
    
    # Look for checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoints_dir):
        if file.endswith('.pth') and ('checkpoint_update_' in file or file == 'best_model.pth'):
            file_path = os.path.join(checkpoints_dir, file)
            checkpoint_files.append((file_path, os.path.getmtime(file_path)))
    
    if not checkpoint_files:
        return None
    
    # Return the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])[0]
    return latest_checkpoint

def load_checkpoint_cpu(model: TagTransformer, 
                       optimizer: optim.Optimizer, 
                       checkpoint_path: str,
                       device: torch.device) -> Tuple[int, int, dict]:
    """Fast checkpoint loading with enhanced error handling"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        updates = checkpoint.get('updates', 0)  # Default to 0 if not present in old checkpoints
        data_paths = checkpoint.get('data_paths', {})  # Default to empty dict if not present
        
        print(f'✓ Checkpoint loaded successfully: {checkpoint_path}')
        print(f'  Epoch: {epoch}, Updates: {updates}, Loss: {loss:.4f}')
        
        if data_paths:
            print(f'  Data paths from checkpoint:')
            for key, path in data_paths.items():
                print(f'    {key}: {path}')
        
        return epoch, updates, data_paths
        
    except Exception as e:
        print(f'✗ Error loading checkpoint {checkpoint_path}: {e}')
        raise

def setup_cpu_environment():
    """Setup aggressive CPU optimizations"""
    
    # Set number of threads
    num_threads = os.cpu_count()
    
    print(f"✓ CPU Cores: {num_threads}")
    print(f"✓ PyTorch threads: {torch.get_num_threads()}")
    print(f"✓ PyTorch interop threads: {torch.get_num_interop_threads()}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Training script for morphological reinflection with automatic CUDA/CPU detection')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (e.g., ./models/checkpoints/best_model.pth)')
    parser.add_argument('--auto_resume', action='store_true', help='Automatically resume from the latest checkpoint in output_dir')
    parser.add_argument('--list_checkpoints', action='store_true', help='List available checkpoints and exit')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory')
    parser.add_argument('--train_src', type=str, default='./10L_90NL/train/run1/train.10L_90NL_1_1.src', help='Training source file path')
    parser.add_argument('--train_tgt', type=str, default='./10L_90NL/train/run1/train.10L_90NL_1_1.tgt', help='Training target file path')
    parser.add_argument('--dev_src', type=str, default='./10L_90NL/dev/run1/dev.10L_90NL_1_1.src', help='Development source file path')
    parser.add_argument('--dev_tgt', type=str, default='./10L_90NL/dev/run1/dev.10L_90NL_1_1.tgt', help='Development target file path')
    parser.add_argument('--test_src', type=str, default='./10L_90NL/test/run1/test.10L_90NL_1_1.src', help='Test source file path (optional)')
    parser.add_argument('--test_tgt', type=str, default='./10L_90NL/test/run1/test.10L_90NL_1_1.tgt', help='Test target file path (optional)')
    args = parser.parse_args()
    
    # Handle list checkpoints option
    if args.list_checkpoints:
        checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            print(f"No checkpoints directory found at: {checkpoints_dir}")
            return
        
        checkpoint_files = []
        for file in os.listdir(checkpoints_dir):
            if file.endswith('.pth'):
                file_path = os.path.join(checkpoints_dir, file)
                try:
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                    epoch = checkpoint.get('epoch', 'Unknown')
                    updates = checkpoint.get('updates', 'Unknown')
                    loss = checkpoint.get('loss', 'Unknown')
                    mtime = os.path.getmtime(file_path)
                    checkpoint_files.append((file, epoch, updates, loss, mtime))
                except Exception as e:
                    checkpoint_files.append((file, 'Error', 'Error', 'Error', os.path.getmtime(file_path)))
        
        if not checkpoint_files:
            print(f"No checkpoint files found in: {checkpoints_dir}")
            return
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x[4], reverse=True)
        
        print(f"Available checkpoints in {checkpoints_dir}:")
        print(f"{'File':<30} {'Epoch':<8} {'Updates':<10} {'Loss':<12} {'Modified'}")
        print("-" * 80)
        for file, epoch, updates, loss, mtime in checkpoint_files:
            mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            print(f"{file:<30} {epoch:<8} {updates:<10} {loss_str:<12} {mtime_str}")
        return
    
    # CPU-optimized configuration - using same hyperparameters as original
    config = {
        'embed_dim': 256,
        'nb_heads': 4,
        'src_hid_size': 1024,
        'src_nb_layers': 4,
        'trg_hid_size': 1024,
        'trg_nb_layers': 4,
        'dropout_p': 0.1,
        'tie_trg_embed': True,
        'label_smooth': 0.1,
        'batch_size': 400,  # Fixed batch size - never change this
        'learning_rate': 0.001,
        'max_epochs': 1000,  # Not used as termination condition anymore
        'max_updates': 10000,  # Primary termination condition
        'warmup_steps': 4000,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'save_every': 10,  # Same as original
        'eval_every': 5,   # Same as original
        'max_length': 100,
        'gradient_accumulation_steps': 2,  # Same as original
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Validate batch size is fixed at 400
    assert config['batch_size'] == 400, f"Batch size must be 400, got {config['batch_size']}"
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    device = DEVICE
    print(f'Using device: {device}')
    
    # Check if CUDA is available and provide appropriate information
    if torch.cuda.is_available():
        print(f'CUDA detected: {torch.cuda.get_device_name(0)}')
        print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        print('Using CUDA optimizations')
    else:
        print('CUDA not available, using CPU optimizations')
        # Setup CPU environment only when using CPU
        setup_cpu_environment()
    
    # Initialize resume variables (will be set after model/optimizer creation)
    start_epoch = 0
    start_updates = 0
    resume_checkpoint = None
    checkpoint_data_paths = {}
    
    # Data file paths - now configurable via command line
    train_src = args.train_src
    train_tgt = args.train_tgt
    dev_src = args.dev_src
    dev_tgt = args.dev_tgt
    test_src = args.test_src
    test_tgt = args.test_tgt
    
    # Check if data paths match checkpoint (if resuming)
    if resume_checkpoint and checkpoint_data_paths:
        current_data_paths = {
            'train_src': train_src,
            'train_tgt': train_tgt,
            'dev_src': dev_src,
            'dev_tgt': dev_tgt,
            'test_src': test_src,
            'test_tgt': test_tgt
        }
        
        mismatched_paths = []
        for key, checkpoint_path in checkpoint_data_paths.items():
            if key in current_data_paths and current_data_paths[key] != checkpoint_path:
                mismatched_paths.append(f"  {key}: checkpoint='{checkpoint_path}' vs current='{current_data_paths[key]}'")
        
        if mismatched_paths:
            print(f"\n⚠️  WARNING: Data paths don't match checkpoint!")
            print("Mismatched paths:")
            for mismatch in mismatched_paths:
                print(mismatch)
            print("This may cause training inconsistencies. Consider using the same data files.")
            print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
            time.sleep(5)
        else:
            print("✓ Data paths match checkpoint - resuming with same data files")
    
    # Print data paths being used
    print(f"Training data:")
    print(f"  Source: {train_src}")
    print(f"  Target: {train_tgt}")
    print(f"Development data:")
    print(f"  Source: {dev_src}")
    print(f"  Target: {dev_tgt}")
    if test_src and test_tgt:
        print(f"Test data:")
        print(f"  Source: {test_src}")
        print(f"  Target: {test_tgt}")
    
    # Build vocabulary efficiently
    print("Building vocabulary...")
    src_vocab = build_vocabulary([train_src, dev_src])
    tgt_vocab = build_vocabulary([train_tgt, dev_tgt])
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Create datasets
    train_dataset = MorphologicalDataset(train_src, train_tgt, src_vocab, tgt_vocab, config['max_length'])
    dev_dataset = MorphologicalDataset(dev_src, dev_tgt, src_vocab, tgt_vocab, config['max_length'])
    
    # Create test dataset if test paths are provided
    test_dataset = None
    test_loader = None
    if test_src and test_tgt and os.path.exists(test_src) and os.path.exists(test_tgt):
        test_dataset = MorphologicalDataset(test_src, test_tgt, src_vocab, tgt_vocab, config['max_length'])
        test_loader = create_simple_dataloader(test_dataset, config, src_vocab, tgt_vocab, device)
        print(f"✓ Test dataset created with {len(test_dataset)} samples")
    else:
        print("⚠ Test dataset not created (files not found or paths not provided)")
    
    # Create dataloaders with appropriate device settings
    train_loader = create_simple_dataloader(train_dataset, config, src_vocab, tgt_vocab, device)
    dev_loader = create_simple_dataloader(dev_dataset, config, src_vocab, tgt_vocab, device)
    
    # Create CPU-optimized model
    model = create_cpu_optimized_model(config, src_vocab, tgt_vocab)
    model = model.to(device)
    
    # Count parameters
    total_params = model.count_nb_params()
    print(f'Total parameters: {total_params:,}')
    
    # Create optimizer with maximum speed settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=True,  # Use foreach implementation
    )
    
    # Learning rate scheduler - FIXED: now uses global_step instead of epoch
    def lr_lambda(step):
        if step < config['warmup_steps']:
            return float(step) / float(max(1, config['warmup_steps']))
        progress = (step - config['warmup_steps']) / (config['max_updates'] - config['warmup_steps'])
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified (after model and optimizer are created)
    if args.resume:
        # Manual resume from specified checkpoint
        resume_checkpoint = args.resume
        print(f"Resuming from specified checkpoint: {resume_checkpoint}")
        start_epoch, start_updates, checkpoint_data_paths = load_checkpoint_cpu(model, optimizer, resume_checkpoint, device)
    elif args.auto_resume:
        # Auto-resume from latest checkpoint
        resume_checkpoint = find_latest_checkpoint(args.output_dir)
        if resume_checkpoint:
            print(f"Auto-resuming from latest checkpoint: {resume_checkpoint}")
            start_epoch, start_updates, checkpoint_data_paths = load_checkpoint_cpu(model, optimizer, resume_checkpoint, device)
        else:
            print("No checkpoint found for auto-resume, starting from scratch")
    else:
        print("Starting training from scratch")
    
    # Training loop - now based on max_updates instead of max_epochs
    best_val_loss = float('inf')
    global_step = start_updates
    updates = start_updates
    epoch = start_epoch
    
    print(f"\nStarting training with {len(train_loader)} batches per epoch")
    print(f"Batch size: {config['batch_size']} (FIXED)")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Max updates: {config['max_updates']}")
    print(f"Total training samples: {len(train_loader) * config['batch_size']}")
    print(f"Note: '96 batches per epoch' means 96 batches of size 400 each, not batch size 96")
    
    if resume_checkpoint:
        print(f"\n🔄 RESUMING TRAINING:")
        print(f"  From checkpoint: {resume_checkpoint}")
        print(f"  Starting epoch: {start_epoch}")
        print(f"  Starting updates: {start_updates}")
        print(f"  Remaining updates: {config['max_updates'] - start_updates}")
    else:
        print(f"\n🚀 STARTING FRESH TRAINING")
    
    while updates < config['max_updates']:
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch_cpu(
            model, train_loader, optimizer, device, epoch, config
        )
        
        # Update learning rate using global step (not epoch) - FIXED!
        scheduler.step(int(global_step))
        current_lr = scheduler.get_last_lr()[0]
        
        # Count updates for this epoch
        epoch_updates = len(train_loader)
        updates += epoch_updates
        global_step += epoch_updates
        
        # Validate based on updates, not epochs
        if updates % (config['eval_every'] * epoch_updates) < epoch_updates:
            val_loss = validate_cpu(model, dev_loader, device, config)
            
            print(f'Update {updates}/{config["max_updates"]} (Epoch {epoch}): Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                data_paths = {
                    'train_src': train_src,
                    'train_tgt': train_tgt,
                    'dev_src': dev_src,
                    'dev_tgt': dev_tgt,
                    'test_src': test_src,
                    'test_tgt': test_tgt
                }
                save_checkpoint_cpu(
                    model, optimizer, epoch, val_loss,
                    os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'),
                    updates, data_paths
                )
        else:
            print(f'Update {updates}/{config["max_updates"]} (Epoch {epoch}): Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}')
        
        # Save checkpoint based on updates, not epochs
        if updates % (config['save_every'] * epoch_updates) < epoch_updates:
            data_paths = {
                'train_src': train_src,
                'train_tgt': train_tgt,
                'dev_src': dev_src,
                'dev_tgt': dev_tgt,
                'test_src': test_src,
                'test_tgt': test_tgt
            }
            save_checkpoint_cpu(
                model, optimizer, epoch, train_loss,
                os.path.join(args.output_dir, 'checkpoints', f'checkpoint_update_{updates}.pth'),
                updates, data_paths
            )
        
        epoch_time = time.time() - epoch_start_time
        samples_per_sec = len(train_loader) * config['batch_size'] / epoch_time
        
        print(f'Epoch {epoch} completed in {epoch_time:.2f}s ({samples_per_sec:.0f} samples/sec)')
        print(f'Progress: {updates}/{config["max_updates"]} updates ({100*updates/config["max_updates"]:.1f}%)')
        
        # Check if we've reached max updates
        if updates >= config['max_updates']:
            print(f'Reached maximum updates ({config["max_updates"]}), stopping training')
            break
        
        epoch += 1
        
        # Clear memory periodically
        gc.collect()
    
    # Save final model
    data_paths = {
        'train_src': train_src,
        'train_tgt': train_tgt,
        'dev_src': dev_src,
        'dev_tgt': dev_tgt,
        'test_src': test_src,
        'test_tgt': test_tgt
    }
    save_checkpoint_cpu(
        model, optimizer, epoch, train_loss,
        os.path.join(args.output_dir, 'checkpoints', 'final_model.pth'),
        updates, data_paths
    )
    
    # Final evaluation on test set if available
    if test_loader is not None:
        print("\n" + "="*50)
        print("FINAL TEST EVALUATION")
        print("="*50)
        
        test_loss = evaluate_test_cpu(model, test_loader, device, config)
        print(f"Final Test Loss: {test_loss:.4f}")
        
        # Save test results
        test_results = {
            'test_loss': test_loss,
            'final_epoch': epoch,
            'final_updates': updates,
            'final_train_loss': train_loss,
            'best_val_loss': best_val_loss
        }
        
        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Test results saved to: {os.path.join(args.output_dir, 'test_results.json')}")
    
    print('CPU-optimized training completed!')

