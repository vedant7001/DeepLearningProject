# Run DeepLearningProject on Google Colab

This document contains code and instructions for running the DeepLearningProject repository on Google Colab with GPU acceleration. Follow these steps to create a new notebook and run your project.

## Instructions:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Copy and paste the code cells below into your notebook
4. Set the runtime type to GPU: Click on "Runtime" > "Change runtime type" > Select "GPU"
5. Run the cells in sequence

---

## Step 1: Check for GPU availability

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
```

## Step 2: Mount Google Drive (Optional)

```python
from google.colab import drive
drive.mount('/content/drive')

# Create a directory for our project in Google Drive
!mkdir -p /content/drive/MyDrive/DeepLearningProject
```

## Step 3: Clone the repository and install dependencies

```python
# Clone the repository
!git clone https://github.com/vedant7001/DeepLearningProject.git
%cd DeepLearningProject

# Install dependencies
!pip install -q torch transformers datasets tqdm tensorboard
```

## Step 4: Set up the Python path

```python
import sys
import os

# Add the repository root to the Python path
repo_path = os.path.abspath('.')
if repo_path not in sys.path:
    sys.path.append(repo_path)
    
print(f"Repository path '{repo_path}' added to Python path")
```

## Step 5: Run the training script

```python
# Run the LSTM model training with GPU acceleration
!python Result/main.py train \
    --model_type lstm \
    --embed_size 128 \
    --hidden_size 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --learning_rate 1e-3 \
    --num_epochs 5 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --output_dir /content/drive/MyDrive/DeepLearningProject/model_outputs
```

## Step 6: Evaluate the trained model

```python
# Create a model config file for evaluation
import json

model_configs = {
    "lstm": {
        "model_type": "lstm",
        "model_path": "/content/drive/MyDrive/DeepLearningProject/model_outputs/lstm/lstm_model_best.pt",
        "embed_size": 128,
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.3
    }
}

config_path = "model_configs.json"
with open(config_path, "w") as f:
    json.dump(model_configs, f, indent=4)
    
print(f"Model config saved to {config_path}")
```

```python
# Run evaluation
!python Result/main.py evaluate \
    --model_configs model_configs.json \
    --data_split val \
    --batch_size 32 \
    --save_results \
    --output_dir /content/drive/MyDrive/DeepLearningProject/evaluation_results
```

## Step 7: Make predictions with the trained model

```python
# Run prediction in interactive mode
# Note: This might not work well in Colab due to input limitations
# You can modify this to load questions from a file instead
!python Result/main.py predict \
    --model_type lstm \
    --model_path /content/drive/MyDrive/DeepLearningProject/model_outputs/lstm/lstm_model_best.pt \
    --mode interactive
```

## Additional: Train with different models

```python
# Train the attention-based LSTM model
!python Result/main.py train \
    --model_type attn \
    --embed_size 128 \
    --hidden_size 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --learning_rate 1e-3 \
    --num_epochs 5 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --output_dir /content/drive/MyDrive/DeepLearningProject/model_outputs
```

```python
# Train the transformer model
!python Result/main.py train \
    --model_type transformer \
    --model_size base \
    --dropout 0.1 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --output_dir /content/drive/MyDrive/DeepLearningProject/model_outputs
``` 