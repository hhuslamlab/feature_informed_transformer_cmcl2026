---
title: Morphological Transformer Training
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
short_description: Train morphological reinflection models using TagTransformer
---

# Morphological Transformer Training

Train and test morphological reinflection models using TagTransformer architecture on Hugging Face Spaces.

## 🚀 Features

- **Training Interface**: Train models on different datasets (10L_90NL, 50L_50NL, 90L_10NL)
- **Model Management**: View and manage trained models
- **Testing Interface**: Test trained models with custom inputs
- **Monitoring**: Integration with Weights & Biases for experiment tracking
- **Cloud Optimized**: Designed for Hugging Face Spaces infrastructure

## 📊 Datasets

- **10L_90NL**: 10% labeled, 90% non-labeled data
- **50L_50NL**: 50% labeled, 50% non-labeled data
- **90L_10NL**: 90% labeled, 10% non-labeled data

## 🔧 Setup

### Environment Variables

Set these environment variables in your Space settings:

- `HF_TOKEN`: Your Hugging Face token for model upload
- `WANDB_TOKEN`: Your Weights & Biases token (optional)
- `WANDB_PROJECT`: Project name for experiment tracking

### Data Mounting

Mount your data directory to `/data` with the following structure:

```
/data/
├── 10L_90NL/
│   ├── train/run1/
│   ├── dev/run1/
│   └── test/run1/
├── 50L_50NL/
│   ├── train/run1/
│   ├── dev/run1/
│   └── test/run1/
└── 90L_10NL/
    ├── train/run1/
    ├── dev/run1/
    └── test/run1/
```

## 🎯 Usage

1. **Training**: Go to the Training tab, configure parameters, and start training
2. **Monitoring**: Watch training progress in the logs and Weights & Biases
3. **Testing**: Use the Testing tab to test your trained models
4. **Model Management**: View available models in the Models tab

## 📈 Training Configuration

The training uses optimized settings for cloud infrastructure:

- **Batch Size**: 32 (GPU) / 16 (CPU)
- **Learning Rate**: 0.001
- **Max Epochs**: 100
- **Gradient Accumulation**: 4 steps
- **Mixed Precision**: Enabled on GPU

## 🔍 Model Architecture

The TagTransformer uses:

- **Encoder-Decoder Architecture**: Transformer-based sequence-to-sequence model
- **Feature Embeddings**: Special embeddings for morphological features
- **Positional Encoding**: Custom positional encoding for character sequences
- **Label Smoothing**: Improved training stability

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{morphological-transformer,
  title={Morphological Transformer for Reinflection},
  author={Your Name},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/spaces/your-username/morphological-transformer}}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.







