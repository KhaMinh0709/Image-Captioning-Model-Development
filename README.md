# Image Captioning Model Development

## Overview

This project implements a deep learning-based image captioning system that combines computer vision and natural language processing to automatically generate descriptive captions for images. The model uses a Vision Encoder-Decoder architecture with attention mechanisms to produce accurate and contextually relevant image descriptions.

## Project Architecture

### 1. Encoder
- **Component**: ResNet-50 pre-trained on ImageNet
- **Function**: Extracts visual features from input images
- **Process**: 
  - Removes fully connected layers from ResNet-50
  - Projects extracted features to embedding space (512-dimensional)
  - Freezes ResNet weights to preserve pre-trained knowledge

### 2. Decoder
- **Component**: LSTM with Multi-Head Attention mechanism
- **Function**: Converts visual features into natural language captions
- **Features**:
  - Word embedding layer for caption token encoding
  - Multi-layer LSTM for sequence generation
  - Multi-head attention (8 heads) for dynamic feature weighting
  - Greedy decoding strategy for caption generation

## Dataset

The project uses two publicly available image captioning datasets:

### Flickr30K Dataset
- **Size**: 30,000 images with 5 captions each
- **Source**: Kaggle (nunenuh/flickr30k)
- **Content**: General web images with descriptive captions

### COCO Dataset
- **Size**: 80,000 training + 40,000 validation images
- **Source**: Kaggle (nikhil7280/coco-image-caption)
- **Content**: Complex scenes with multiple objects and detailed descriptions
- **Format**: JSON annotations with image metadata

**Combined Dataset Statistics:**
- Total samples: ~420,000 image-caption pairs
- Train/Val/Test split: 80% / 10% / 10%

## Data Preprocessing

### Image Processing
- Resize to 224×224 pixels (ResNet input size)
- Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Training augmentation**: Random horizontal flips, rotation (±20°)
- **Validation/Test**: No augmentation applied

### Caption Processing
- Tokenization using BERT tokenizer (bert-base-uncased)
- Special tokens: [CLS] (start), [SEP] (end), [PAD] (padding)
- Maximum caption length: 50 tokens
- Automatic padding/truncation to fixed length

## Model Specifications

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Embedding Size | 512 |
| Hidden Size | 512 |
| Vocabulary Size | 30,522 (BERT vocab) |
| Batch Size | 32 |
| Learning Rate | 1×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ |
| Number of Epochs | 2 |
| Max Caption Length | 50 tokens |

### Training Configuration
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate Scheduler**: ReduceLROnPlateau (patience=2, factor=0.1, min_lr=1×10⁻⁶)
- **Early Stopping**: Patience=5 epochs
- **Hardware**: GPU acceleration (CUDA)

## Training Process

### Data Loading Pipeline
1. Images and captions loaded from local paths
2. Custom `CaptionDataset` class handles image loading and caption encoding
3. DataLoader with collate function manages batching:
   - Stacks images into batch tensors (32, 3, 224, 224)
   - Pads captions to consistent length
   - Custom collate_fn handles heterogeneous sequence lengths

### Training Loop
- Forward pass: Images → Encoder → Features → Decoder → Caption predictions
- Loss calculation on flattened output and caption tensors
- Backward propagation with gradient accumulation
- Metrics tracked: Training loss, perplexity, validation loss, perplexity
- TensorBoard logging for visualization

### Monitoring
- Per-epoch validation on separate validation set
- Model checkpoint saving when validation loss improves
- Learning rate adjustment based on validation loss plateau
- Early stopping prevents overfitting

## Evaluation Metrics

The model is evaluated using **BLEU (Bilingual Evaluation Understudy)** scores:

### BLEU Score Variants
- **BLEU-1**: Unigram overlap (word-level precision)
- **BLEU-2**: Bigram overlap (two-word phrase matching)
- **BLEU-3**: Trigram overlap (three-word phrase matching)
- **BLEU-4**: 4-gram overlap (sentence structure similarity)

### Evaluation Results
Average BLEU scores computed across entire test set with standard smoothing function applied.

## Key Features

✅ **Pre-trained Backbone**: ResNet-50 for robust feature extraction  
✅ **Attention Mechanism**: Multi-head attention for focus on relevant image regions  
✅ **LSTM Decoder**: Effective sequence modeling for caption generation  
✅ **Flexible Dataset Support**: Handles both Flickr30K and COCO formats  
✅ **Data Augmentation**: Improves model robustness during training  
✅ **Early Stopping**: Prevents overfitting with patience-based monitoring  
✅ **Model Checkpointing**: Saves best model state based on validation performance  

## Usage

### Installation Requirements
```python
pip install torch torchvision transformers pillow kagglehub pandas matplotlib tqdm
pip install pycocotools ipywidgets
```

### Inference on New Images

```python
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import BertTokenizer

# Load model and tokenizer
model = ImageCaptioningModel(embed_size=512, hidden_size=512, vocab_size=30522)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare image
image = Image.open('path/to/image.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = transform(image).unsqueeze(0)

# Generate caption
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image = image.to(device)
model.to(device)

with torch.no_grad():
    features = model.encoder(image)
    caption_ids = model.decoder.greedy_decode(features, max_length=50, 
                                             start_token=tokenizer.cls_token_id,
                                             end_token=tokenizer.sep_token_id)
    caption = tokenizer.decode(caption_ids, skip_special_tokens=True)
    print(f"Generated Caption: {caption}")
```

## Project Structure

```
image_captioning/
├── image_captioning.ipynb      # Main Jupyter notebook with full pipeline
├── README.md                    # This file
└── best_model.pth              # Saved model weights
```

## Notebook Sections

1. **Library Imports**: Essential dependencies and frameworks
2. **Dataset Loading**: Flickr30K and COCO dataset download and processing
3. **Data Splitting**: Train/validation/test split (80/10/10)
4. **Data Preparation**: Custom Dataset class and DataLoader setup
5. **Model Architecture**: Encoder and Decoder class definitions
6. **Model Initialization**: Combining components into integrated model
7. **Training**: Main training loop with validation and early stopping
8. **Model Evaluation**: BLEU score computation on test set
9. **Inference**: Caption generation on test images
10. **Model Saving**: Checkpoint management and state dict serialization
11. **Model Loading**: Restoring trained model for inference

## Results

The model demonstrates effective image-to-text translation capabilities:
- Successfully generates coherent, contextually relevant captions
- Captures objects, actions, and scene relationships
- Generalizes well to unseen images from combined training data
- BLEU scores show reasonable overlap with ground truth captions

## Limitations and Future Work

### Current Limitations
- Limited vocabulary to BERT-base tokenizer (30K tokens)
- Fixed maximum caption length (50 tokens) may truncate longer descriptions
- Greedy decoding may miss optimal caption alternatives
- Single model architecture without ensemble methods

### Future Improvements
- Implement beam search for better caption quality
- Multi-head attention visualization for interpretability
- Fine-tuning of ResNet encoder weights
- Larger language model integration (GPT-2, GPT-3)
- Custom tokenizer training for domain-specific vocabulary
- Bidirectional encoder architectures
- Graph-based attention for scene understanding
- Evaluation with additional metrics (METEOR, CIDEr, ROUGE)

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation
- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers

## Author

**Student ID**: 21124661  
**Course**: Deep Learning Project  
**Institution**: University of Information Technology

## License

This project is provided for educational purposes. Dataset usage must comply with respective dataset licenses (Flickr30K and COCO).