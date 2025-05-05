# Question Answering System with Multiple Architectures

This project implements and compares three different encoder-decoder architectures for Question Answering using the SQuAD v1.1 dataset.

## Architectures

The system implements three different encoder-decoder architectures:

1. **LSTM Encoder-Decoder (without Attention)**:
   - BiLSTM encoder to process the combined context and question.
   - Unidirectional LSTM decoder to generate the answer sequence.
   - Teacher forcing during training.
   - Token-level cross-entropy loss.

2. **LSTM Encoder-Decoder (with Attention)**:
   - Extends the first model by adding Bahdanau attention.
   - Computes attention weights between the decoder state and encoder outputs.
   - Uses context vectors to inform the decoder's output at each time step.
   - Includes visualization of attention weights.

3. **Transformer-based Encoder-Decoder**:
   - Implements a Transformer model following the "Attention Is All You Need" architecture.
   - Uses self-attention, multi-head attention, positional encoding, and encoder-decoder attention.
   - Includes masking and layer normalization.

## Features

- Uses Hugging Face's tokenizer for preprocessing.
- Implements padding, masking, and batching.
- Supports training, validation, and inference.
- Tracks BLEU, ROUGE-L, and F1 scores.
- Logs loss curves and accuracy plots.
- Saves predictions and metrics.
- Visualization of attention weights.
- Comprehensive results analysis and model comparison.
- Interactive dashboards with comparative examples.
- Failure analysis with best/worst prediction visualization.
- Specialized attention visualization for different model types.
- Experiment management with automated directory structure.
- Batch experiment runner with logging and error handling.

## Project Structure

```
├── data/
│   └── squad_preprocessing.py  # SQuAD dataset preprocessing
├── models/
│   ├── lstm_model.py           # LSTM Encoder-Decoder without attention
│   ├── attn_model.py           # LSTM Encoder-Decoder with attention
│   └── transformer_qa.py       # Transformer Encoder-Decoder
├── training/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── predict.py              # Prediction script
├── utils/
│   ├── metrics.py              # Evaluation metrics
│   ├── tokenization.py         # Tokenization utilities
│   ├── visualization.py        # General visualization utilities
│   └── attention_visualization.py # Specialized attention visualization 
├── results/                    # Model checkpoints and results
├── plots/                      # Training plots and attention visualizations
├── main.py                     # Main entry point
├── results_analysis.py         # Comprehensive results analysis 
├── results_organizer.py        # Directory structure creator
├── run_qa_experiment.py        # Experiment runner
└── requirements.txt            # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train a model:

```bash
python main.py train --model_type <lstm|attn|transformer> [options]
```

Example:
```bash
python main.py train --model_type attn --num_epochs 10 --save_attention
```

Options:
- `--model_type`: Type of model to train (lstm, attn, transformer)
- `--tokenizer_name`: Name of the Hugging Face tokenizer (default: bert-base-uncased)
- `--output_dir`: Directory to save model and logs (default: results)
- `--train_batch_size`: Batch size for training (default: 16)
- `--num_epochs`: Number of epochs to train (default: 10)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--save_attention`: Whether to save attention weights

### Evaluation

To evaluate one or more models:

```bash
python main.py evaluate --model_configs <path_to_config_file> [options]
```

Example:
```bash
python main.py evaluate --model_configs model_configs.json --save_results
```

The model configuration file should be a JSON file with the following structure:
```json
[
  {
    "name": "LSTM Model",
    "model_type": "lstm",
    "model_path": "results/lstm/lstm_encoder_decoder_best.pt"
  },
  {
    "name": "LSTM with Attention",
    "model_type": "attn",
    "model_path": "results/attn/lstm_encoder_attn_decoder_best.pt"
  },
  {
    "name": "Transformer",
    "model_type": "transformer",
    "model_path": "results/transformer/transformer_base_best.pt",
    "model_size": "base"
  }
]
```

### Prediction

For interactive prediction:

```bash
python main.py predict --model_type <lstm|attn|transformer> --model_path <path_to_model> --mode interactive
```

For prediction from a file:

```bash
python main.py predict --model_type <lstm|attn|transformer> --model_path <path_to_model> --mode file --input_file <input_file> --output_file <output_file>
```

Example:
```bash
python main.py predict --model_type transformer --model_path results/transformer/transformer_base_best.pt --mode interactive --visualize_attention
```

### Model Comparison

To compare multiple models:

```bash
python main.py compare --results_dir <directory_with_results> --output_dir <output_directory>
```

### Experiment Management

For complete experiment management (setup, training, evaluation, and analysis):

```bash
python run_qa_experiment.py [options]
```

Options:
- `--experiment_name`: Name for the experiment (optional)
- `--base_dir`: Base directory for the project
- `--setup_only`: Only set up the experiment without running it
- `--analyze_only`: Only analyze results without running the experiment
- `--visualize_attention`: Generate attention visualizations
- `--existing_experiment`: Path to an existing experiment directory (for analyze_only)

Example:
```bash
# Set up an experiment structure
python run_qa_experiment.py --experiment_name qa_benchmark --setup_only

# Run a complete experiment with attention visualization
python run_qa_experiment.py --experiment_name qa_benchmark --visualize_attention

# Analyze results from an existing experiment
python run_qa_experiment.py --analyze_only --existing_experiment results/qa_benchmark_20240520_123456
```

### Results Analysis

To analyze results from a completed experiment:

```bash
python results_analysis.py --results_dir <results_directory> --output_dir <output_directory>
```

This will generate:
- Comprehensive metrics tables and plots
- Comparative examples across models
- Failure analysis for each model
- Interactive HTML dashboard 

### Attention Visualization

To create specialized attention visualizations:

```bash
python -m utils.attention_visualization --attention_file <path_to_attention_weights> --model_type <lstm|attn|transformer> --output_dir attention_viz
```

## Metrics

The system evaluates models using the following metrics:

- **Exact Match**: Percentage of predictions that exactly match the ground truth.
- **F1 Score**: Harmonic mean of precision and recall at the token level.
- **BLEU Score**: Measures the precision of n-grams between the prediction and ground truth.
- **ROUGE-L Score**: Measures the longest common subsequence between the prediction and ground truth.

## Visualization

The system generates the following visualizations:

- **Training Loss Curves**: Plot of training and validation loss over epochs.
- **Metric Plots**: Plot of evaluation metrics over epochs.
- **Attention Weights**: Heatmap of attention weights for models with attention.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The SQuAD dataset by Stanford University.
- The Hugging Face team for their transformers library.
- The PyTorch team for the excellent deep learning framework. 