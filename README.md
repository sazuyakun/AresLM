# AresLM - An LLM built locally on a rookie's laptop 🚀

**AresLM** is a from-scratch implementation of a GPT-style Large Language Model (LLM) built for educational purposes and experimentation. This project demonstrates how to build, train, and fine-tune a transformer-based language model using PyTorch, complete with all the fundamental components including attention mechanisms, transformer blocks, and text generation capabilities.

## Project Overview

This repository contains a complete implementation of a GPT-2 style language model, built from the ground up to understand the inner workings of modern LLMs. The project includes:

- **Custom GPT Architecture**: Full implementation of transformer blocks with multi-head attention
- **Data Processing Pipeline**: Text tokenization and embedding systems
- **Training Infrastructure**: Pre-training and fine-tuning capabilities
- **Experimental Notebooks**: Step-by-step Jupyter notebooks exploring each component
- **Text Generation**: Multiple decoding strategies for text generation

## Architecture Components

### Core Components
- **Multi-Head Attention**: Causal self-attention mechanism with configurable heads
- **Transformer Blocks**: Complete transformer architecture with residual connections
- **Feed-Forward Networks**: Position-wise feed-forward layers with GELU activation
- **Layer Normalization**: Custom implementation for stable training
- **Positional Embeddings**: Learned positional encodings for sequence understanding

### Model Specifications
- **Model Size**: GPT-2 124M parameter equivalent
- **Vocabulary Size**: 50,257 tokens (GPT-2 tokenizer)
- **Context Length**: 256 tokens (configurable)
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Transformer Layers**: 12
- **Dropout Rate**: 0.1

## Tech Stack

### Core Dependencies
- **PyTorch** - Deep learning framework
- **TensorFlow** - For loading pre-trained GPT-2 weights
- **tiktoken** - GPT-2/GPT-3 tokenizer
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

### Development & Visualization
- **JupyterLab** - Interactive development environment
- **Matplotlib** - Plotting and visualization
- **tqdm** - Progress bars
- **psutil** - System monitoring

## 📁 Project Structure

```
llm-from-scratch/
├── README.md
├── main.py
├── requirements.txt
├── setup.py
│
├── config/
│   └── config.py
│
├── attention/
│   ├── causal_attention.py
│   └── multi_head_attention.py
│
├── gpt/
│   ├── gpt_model.py
│   ├── transformer_block.py
│   ├── ff.py
│   ├── layer_norm.py
│   └── gelu.py
│
├── data/
│   ├── dataloader.py
│   ├── embedding.py
│   └── the-verdict.txt
│
├── utils/
│   └── utils.py
│
└── experiments/
    ├── 01-data/
    ├── 02-attention/
    ├── 03-LLM/
    ├── 04-pretrain/
    └── 05-finetune/
```

## Getting Started

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm-from-scratch
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install the package in development mode:
   ```bash
   pip install -e .
   ```

3. **Run the main script**:
   ```bash
   python main.py
   ```

### Usage Examples

#### Basic Text Generation
```python
from gpt.gpt_model import GPTModel
from config.config import GPT_CONFIG_124M
from utils.utils import generate_text_simple
import tiktoken

# Initialize model
model = GPTModel(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")

# Generate text
start_context = "Hello, I am"
# ... (see main.py for complete example)
```

#### Exploring Components
The `experiments/` directory contains detailed Jupyter notebooks:
- `01-data/`: Text processing and tokenization
- `02-attention/`: Attention mechanisms step-by-step
- `03-LLM/`: Building the complete GPT model
- `04-pretrain/`: Pre-training on text data
- `05-finetune/`: Fine-tuning for specific tasks

## Experiments & Learning Path

### 1. Data Processing (`experiments/01-data/`)
- Text tokenization with GPT-2 tokenizer
- Creating data loaders for training
- Token embeddings and positional encodings

### 2. Attention Mechanisms (`experiments/02-attention/`)
- Simple self-attention from scratch
- Causal attention for autoregressive generation
- Multi-head attention implementation
- Attention visualization and analysis

### 3. GPT Model (`experiments/03-LLM/`)
- Complete GPT architecture assembly
- Layer normalization and residual connections
- Feed-forward networks with GELU activation
- Model parameter counting and analysis

### 4. Pre-training (`experiments/04-pretrain/`)
- Training loop implementation
- Loss computation and optimization
- Loading pre-trained GPT-2 weights
- Text generation strategies (greedy, temperature sampling)

### 5. Fine-tuning (`experiments/05-finetune/`)
- Fine-tuning for classification tasks
- SMS spam detection example
- Transfer learning techniques
- Performance evaluation

## 🎛️ Configuration

The model configuration is centralized in `config/config.py`:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # GPT-2 vocabulary size
    "context_length": 256,    # Sequence length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of transformer layers
    "drop_rate": 0.1,         # Dropout probability
    "qkv_bias": False         # Bias in attention projections
}
```

## Key Features

- **Educational Focus**: Clear, well-commented code for learning
- **Modular Design**: Each component can be studied independently
- **Compatibility**: Can load pre-trained GPT-2 weights
- **Experimentation**: Comprehensive notebooks for hands-on learning
- **Scalability**: Configurable architecture for different model sizes

## Model Performance

The model supports various configurations:
- **GPT-2 Small (124M)**: 768 dim, 12 layers, 12 heads
- **GPT-2 Medium (355M)**: 1024 dim, 24 layers, 16 heads
- **GPT-2 Large (774M)**: 1280 dim, 36 layers, 20 heads
- **GPT-2 XL (1558M)**: 1600 dim, 48 layers, 25 heads

## Contributing

This project is designed for educational purposes. Feel free to:
- Experiment with different architectures
- Add new features or optimizations
- Improve documentation
- Share your learning experiences

## Learning Resources

This implementation is inspired by and follows best practices from:
- "Build a Large Language Model (From Scratch)" methodology
- Original GPT-2 paper and implementation
- Modern transformer architecture principles
- PyTorch deep learning patterns

---

**Built with ❤️ for learning and understanding the magic behind Large Language Models**

