"""
Results directory organizer for QA models.

This script creates the necessary directory structure for organizing
results and plots from different QA models.
"""

import os
import argparse
import logging
import shutil
from datetime import datetime

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def create_directory_structure(base_dir: str, experiment_name: str = None):
    """
    Create a directory structure for organizing results and plots.
    
    Args:
        base_dir: Base directory for the project.
        experiment_name: Name of the experiment (optional).
    
    Returns:
        Dictionary with paths to different directories.
    """
    # Create timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # If experiment name not provided, use timestamp
    if experiment_name is None:
        experiment_name = f"experiment_{timestamp}"
    else:
        experiment_name = f"{experiment_name}_{timestamp}"
    
    # Create main experiment directory
    experiment_dir = os.path.join(base_dir, "results", experiment_name)
    
    # Define subdirectories
    dirs = {
        "experiment": experiment_dir,
        "models": {
            "lstm": os.path.join(experiment_dir, "models", "lstm"),
            "attn": os.path.join(experiment_dir, "models", "attn"),
            "transformer": os.path.join(experiment_dir, "models", "transformer"),
        },
        "logs": os.path.join(experiment_dir, "logs"),
        "plots": {
            "training": os.path.join(experiment_dir, "plots", "training"),
            "evaluation": os.path.join(experiment_dir, "plots", "evaluation"),
            "attention": os.path.join(experiment_dir, "plots", "attention"),
            "comparison": os.path.join(experiment_dir, "plots", "comparison"),
        },
        "evaluation": {
            "lstm": os.path.join(experiment_dir, "evaluation", "lstm"),
            "attn": os.path.join(experiment_dir, "evaluation", "attn"),
            "transformer": os.path.join(experiment_dir, "evaluation", "transformer"),
            "comparison": os.path.join(experiment_dir, "evaluation", "comparison"),
        },
        "predictions": os.path.join(experiment_dir, "predictions"),
        "analysis": os.path.join(experiment_dir, "analysis"),
        "tensorboard": os.path.join(experiment_dir, "tensorboard"),
    }
    
    # Create directories
    for key, path in dirs.items():
        if isinstance(path, dict):
            for subkey, subpath in path.items():
                os.makedirs(subpath, exist_ok=True)
                logger.info(f"Created directory: {subpath}")
        else:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")
    
    # Create a README in the experiment directory
    readme_path = os.path.join(experiment_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Experiment: {experiment_name}\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Directory Structure\n\n")
        f.write("- `models/` - Model checkpoints for each architecture\n")
        f.write("- `logs/` - Training and evaluation logs\n")
        f.write("- `plots/` - Visualizations of training progress, attention, etc.\n")
        f.write("- `evaluation/` - Evaluation metrics and results\n")
        f.write("- `predictions/` - Model predictions on test data\n")
        f.write("- `analysis/` - In-depth analysis of model performance\n")
        f.write("- `tensorboard/` - TensorBoard logs for training monitoring\n\n")
        f.write("## Models\n\n")
        f.write("1. LSTM Encoder-Decoder\n")
        f.write("2. LSTM Encoder-Decoder with Attention\n")
        f.write("3. Transformer Encoder-Decoder\n\n")
        f.write("## Experiment Notes\n\n")
        f.write("*Add your notes about this experiment here.*\n")
    
    logger.info(f"Created README at {readme_path}")
    
    # Create a configuration file template
    config_path = os.path.join(experiment_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        f.write('{\n')
        f.write('    "experiment_name": "' + experiment_name + '",\n')
        f.write('    "date": "' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '",\n')
        f.write('    "models": {\n')
        f.write('        "lstm": {\n')
        f.write('            "embed_size": 256,\n')
        f.write('            "hidden_size": 512,\n')
        f.write('            "num_layers": 2,\n')
        f.write('            "dropout": 0.3,\n')
        f.write('            "learning_rate": 3e-4,\n')
        f.write('            "batch_size": 16,\n')
        f.write('            "epochs": 10\n')
        f.write('        },\n')
        f.write('        "attn": {\n')
        f.write('            "embed_size": 256,\n')
        f.write('            "hidden_size": 512,\n')
        f.write('            "num_layers": 2,\n')
        f.write('            "dropout": 0.3,\n')
        f.write('            "learning_rate": 3e-4,\n')
        f.write('            "batch_size": 16,\n')
        f.write('            "epochs": 10\n')
        f.write('        },\n')
        f.write('        "transformer": {\n')
        f.write('            "model_size": "base",\n')
        f.write('            "d_model": 512,\n')
        f.write('            "nhead": 8,\n')
        f.write('            "num_encoder_layers": 6,\n')
        f.write('            "num_decoder_layers": 6,\n')
        f.write('            "dim_feedforward": 2048,\n')
        f.write('            "dropout": 0.1,\n')
        f.write('            "learning_rate": 1e-4,\n')
        f.write('            "batch_size": 16,\n')
        f.write('            "epochs": 10\n')
        f.write('        }\n')
        f.write('    },\n')
        f.write('    "tokenizer": "bert-base-uncased",\n')
        f.write('    "dataset": "squad_v1.1",\n')
        f.write('    "seed": 42,\n')
        f.write('    "notes": "Add your experiment notes here."\n')
        f.write('}\n')
    
    logger.info(f"Created configuration template at {config_path}")
    
    # Create a shell script for running the experiment
    script_path = os.path.join(experiment_dir, "run_experiment.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Script to run the complete experiment pipeline\n\n")
        f.write("# Base directory\n")
        f.write(f"BASE_DIR=\"{base_dir}\"\n")
        f.write(f"EXPERIMENT_DIR=\"{experiment_dir}\"\n\n")
        
        f.write("# 1. Train LSTM model\n")
        f.write("echo \"Training LSTM model...\"\n")
        f.write("python $BASE_DIR/main.py train \\\n")
        f.write("    --model_type lstm \\\n")
        f.write("    --tokenizer_name bert-base-uncased \\\n")
        f.write("    --output_dir $EXPERIMENT_DIR/models/lstm \\\n")
        f.write("    --train_batch_size 16 \\\n")
        f.write("    --num_epochs 10 \\\n")
        f.write("    --learning_rate 3e-4 \\\n")
        f.write("    --embed_size 256 \\\n")
        f.write("    --hidden_size 512 \\\n")
        f.write("    --num_layers 2 \\\n")
        f.write("    --dropout 0.3 \\\n")
        f.write("    --seed 42\n\n")
        
        f.write("# 2. Train LSTM with attention model\n")
        f.write("echo \"Training LSTM with attention model...\"\n")
        f.write("python $BASE_DIR/main.py train \\\n")
        f.write("    --model_type attn \\\n")
        f.write("    --tokenizer_name bert-base-uncased \\\n")
        f.write("    --output_dir $EXPERIMENT_DIR/models/attn \\\n")
        f.write("    --train_batch_size 16 \\\n")
        f.write("    --num_epochs 10 \\\n")
        f.write("    --learning_rate 3e-4 \\\n")
        f.write("    --embed_size 256 \\\n")
        f.write("    --hidden_size 512 \\\n")
        f.write("    --num_layers 2 \\\n")
        f.write("    --dropout 0.3 \\\n")
        f.write("    --save_attention \\\n")
        f.write("    --seed 42\n\n")
        
        f.write("# 3. Train Transformer model\n")
        f.write("echo \"Training Transformer model...\"\n")
        f.write("python $BASE_DIR/main.py train \\\n")
        f.write("    --model_type transformer \\\n")
        f.write("    --tokenizer_name bert-base-uncased \\\n")
        f.write("    --output_dir $EXPERIMENT_DIR/models/transformer \\\n")
        f.write("    --train_batch_size 16 \\\n")
        f.write("    --num_epochs 10 \\\n")
        f.write("    --learning_rate 1e-4 \\\n")
        f.write("    --model_size base \\\n")
        f.write("    --dropout 0.1 \\\n")
        f.write("    --save_attention \\\n")
        f.write("    --seed 42\n\n")
        
        f.write("# 4. Create model configs for evaluation\n")
        f.write("echo \"Creating model configs...\"\n")
        f.write("cat > $EXPERIMENT_DIR/model_configs.json << EOL\n")
        f.write("[\n")
        f.write("  {\n")
        f.write("    \"name\": \"LSTM Encoder-Decoder\",\n")
        f.write("    \"model_type\": \"lstm\",\n")
        f.write("    \"model_path\": \"$EXPERIMENT_DIR/models/lstm/lstm_encoder_decoder_best.pt\"\n")
        f.write("  },\n")
        f.write("  {\n")
        f.write("    \"name\": \"LSTM Encoder-Decoder with Attention\",\n")
        f.write("    \"model_type\": \"attn\",\n")
        f.write("    \"model_path\": \"$EXPERIMENT_DIR/models/attn/lstm_encoder_attn_decoder_best.pt\"\n")
        f.write("  },\n")
        f.write("  {\n")
        f.write("    \"name\": \"Transformer Encoder-Decoder\",\n")
        f.write("    \"model_type\": \"transformer\",\n")
        f.write("    \"model_path\": \"$EXPERIMENT_DIR/models/transformer/transformer_base_best.pt\",\n")
        f.write("    \"model_size\": \"base\"\n")
        f.write("  }\n")
        f.write("]\n")
        f.write("EOL\n\n")
        
        f.write("# 5. Evaluate all models\n")
        f.write("echo \"Evaluating all models...\"\n")
        f.write("python $BASE_DIR/main.py evaluate \\\n")
        f.write("    --model_configs $EXPERIMENT_DIR/model_configs.json \\\n")
        f.write("    --tokenizer_name bert-base-uncased \\\n")
        f.write("    --data_split val \\\n")
        f.write("    --batch_size 16 \\\n")
        f.write("    --output_dir $EXPERIMENT_DIR/evaluation/comparison \\\n")
        f.write("    --save_results \\\n")
        f.write("    --save_attention\n\n")
        
        f.write("# 6. Run analysis on the results\n")
        f.write("echo \"Analyzing results...\"\n")
        f.write("python $BASE_DIR/results_analysis.py \\\n")
        f.write("    --results_dir $EXPERIMENT_DIR/evaluation/comparison \\\n")
        f.write("    --output_dir $EXPERIMENT_DIR/analysis\n\n")
        
        f.write("echo \"Experiment completed! See results in $EXPERIMENT_DIR/analysis\"\n")
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    logger.info(f"Created experiment run script at {script_path}")
    
    return dirs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create directory structure for QA results")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory for the project")
    parser.add_argument("--experiment_name", type=str,
                        help="Name of the experiment (optional)")
    
    args = parser.parse_args()
    
    # Create directory structure
    dirs = create_directory_structure(args.base_dir, args.experiment_name)
    
    logger.info(f"Directory structure created at {dirs['experiment']}")
    logger.info(f"Use the run_experiment.sh script to start the experiment")


if __name__ == "__main__":
    main() 