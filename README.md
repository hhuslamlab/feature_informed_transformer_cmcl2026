# Morphological Transformer Training on Hugging Face

This repository contains code for training morphological reinflection models using TagTransformer architecture on Hugging Face infrastructure.

## 🚀 Quick Start

### Local Training
```bash
# Install dependencies
uv add huggingface_hub transformers datasets wandb

# Login to Hugging Face
uv run huggingface-cli login

# Train a model
uv run python scripts/train_huggingface.py \
    --model_name "your-username/morphological-transformer" \
    --train_src "./10L_90NL/train/run1/train.10L_90NL_1_1.src" \
    --train_tgt "./10L_90NL/train/run1/train.10L_90NL_1_1.tgt" \
    --dev_src "./10L_90NL/dev/run1/dev.10L_90NL_1_1.src" \
    --dev_tgt "./10L_90NL/dev/run1/dev.10L_90NL_1_1.tgt" \
    --wandb_project "morphological-transformer" \
    --upload_model
```

### Cloud Training on Hugging Face Spaces

1. **Create a new Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Create a new Space with "Docker" SDK
   - Set hardware to "GPU" for faster training

2. **Configure Environment Variables**:
   ```bash
   HF_TOKEN=your_huggingface_token
   MODEL_NAME=your-username/morphological-transformer
   DATASET_NAME=10L_90NL
   RUN_NUMBER=1
   WANDB_PROJECT=morphological-transformer-cloud
   ```

3. **Upload your data**:
   - Mount your data directory to `/data`
   - Ensure data structure matches expected format

4. **Start training**:
   - The Space will automatically start training when launched
   - Monitor progress through logs and Weights & Biases

## 📁 Project Structure

```
feature_invariant_transformer/
├── scripts/
│   ├── train_huggingface.py      # Local HF training
│   ├── hf_cloud_training.py      # Cloud training script
│   ├── hf_model_config.py        # HF model configuration
│   ├── train_all_hf.py          # Batch training script
│   ├── transformer.py            # TagTransformer model
│   └── morphological_dataset.py  # Dataset handling
├── requirements_hf.txt           # HF dependencies
├── Dockerfile                    # Docker configuration
├── README.md                     # This file
└── data/                        # Training data
    ├── 10L_90NL/
    ├── 50L_50NL/
    └── 90L_10NL/
```

## 🔧 Configuration

### Environment Variables for Cloud Training

- `HF_TOKEN`: Your Hugging Face token for model upload
- `MODEL_NAME`: Name for your model on HF Hub
- `DATASET_NAME`: Dataset to train (10L_90NL, 50L_50NL, 90L_10NL)
- `RUN_NUMBER`: Run number (1, 2, 3)
- `WANDB_PROJECT`: Weights & Biases project name

### Model Configuration

The cloud training script uses optimized settings for cloud infrastructure:

- **Batch Size**: 32 (GPU) / 16 (CPU)
- **Learning Rate**: 0.001
- **Max Epochs**: 100 (reduced for cloud)
- **Max Updates**: 5000 (reduced for cloud)
- **Gradient Accumulation**: 4 steps
- **Mixed Precision**: Enabled on GPU

## 📊 Training Options

### 1. Local Training
Use `scripts/train_huggingface.py` for local training with full control over hyperparameters.

### 2. Cloud Training
Use `scripts/hf_cloud_training.py` for training on Hugging Face Spaces with optimized cloud settings.

### 3. Batch Training
Use `scripts/train_all_hf.py` to train all datasets and runs automatically.

## 🎯 Datasets

The model supports three dataset configurations:

- **10L_90NL**: 10% labeled, 90% non-labeled data
- **50L_50NL**: 50% labeled, 50% non-labeled data  
- **90L_10NL**: 90% labeled, 10% non-labeled data

Each dataset has 3 runs with different random splits.

## 📈 Monitoring

### Weights & Biases
Training progress is automatically logged to Weights & Biases when configured:

- Training/validation loss
- Learning rate schedule
- Model parameters
- Training time per epoch

### Hugging Face Hub
Models are automatically uploaded to the Hugging Face Hub with:

- Model weights
- Configuration files
- Vocabulary files
- Training arguments
- Model card

## 🚀 Deployment

### Hugging Face Spaces
1. Create a new Space with Docker SDK
2. Set hardware to GPU for faster training
3. Configure environment variables
4. Upload your data
5. Launch the Space

### Local Deployment
1. Install dependencies
2. Configure data paths
3. Run training script
4. Upload models to HF Hub

## 🔍 Model Usage

After training, your models will be available on the Hugging Face Hub:

```python
from transformers import AutoModel, AutoTokenizer

# Load your trained model
model = AutoModel.from_pretrained("your-username/morphological-transformer")
tokenizer = AutoTokenizer.from_pretrained("your-username/morphological-transformer")

# Use for inference
input_text = "example input"
output = model.generate(input_text)
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{morphological-transformer,
  title={Morphological Transformer for Reinflection},
  author={Your Name},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/your-username/morphological-transformer}}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.