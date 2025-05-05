#!/usr/bin/env python
# coding: utf-8

# # Run DeepLearningProject on Google Colab
# 
# This notebook will help you run the DeepLearningProject repository on Google Colab with GPU acceleration. Follow the steps below to get started.

# ## Step 1: Check for GPU availability
# 
# First, let's make sure we have GPU acceleration enabled. In Colab, you can change the runtime type by going to Runtime > Change runtime type and selecting GPU as the hardware accelerator.

# In[ ]:

import torch
print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)

# ## Step 2: Mount Google Drive (Optional)
# 
# Mounting Google Drive allows you to save trained models and results persistently. This is optional but recommended.

# In[ ]:

from google.colab import drive
drive.mount('/content/drive')

# Create a directory for our project in Google Drive
!mkdir -p /content/drive/MyDrive/DeepLearningProject

# ## Step 3: Clone the repository and install dependencies
# 
# This will clone your repository and install all the required dependencies.

# In[ ]:

# Clone the repository
!git clone https://github.com/vedant7001/DeepLearningProject.git
%cd DeepLearningProject

# Install dependencies
!pip install -q torch transformers datasets tqdm tensorboard

# ## Step 4: Set up the Python path
# 
# This ensures that Python can find all the modules in your repository.

# In[ ]:

import sys
import os

# Add the repository root to the Python path
repo_path = os.path.abspath('.')
if repo_path not in sys.path:
    sys.path.append(repo_path)
    
print(f"Repository path '{repo_path}' added to Python path")

# ## Step 5: Run the training script
# 
# Now we can run the training script with GPU acceleration. You can modify the parameters as needed.

# In[ ]:

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

# ## Step 6: Evaluate the trained model
# 
# After training, you can evaluate your model on the validation or test set.

# In[ ]:

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

# In[ ]:

# Run evaluation
!python Result/main.py evaluate \
    --model_configs model_configs.json \
    --data_split val \
    --batch_size 32 \
    --save_results \
    --output_dir /content/drive/MyDrive/DeepLearningProject/evaluation_results

# ## Step 7: Make predictions with the trained model
# 
# You can use the trained model to make predictions in interactive mode.

# In[ ]:

# Run prediction in interactive mode
# Note: This might not work well in Colab due to input limitations
# You can modify this to load questions from a file instead
!python Result/main.py predict \
    --model_type lstm \
    --model_path /content/drive/MyDrive/DeepLearningProject/model_outputs/lstm/lstm_model_best.pt \
    --mode interactive

# ## Additional: Train with different models
# 
# You can also train the attention-based LSTM model or the transformer model by changing the `model_type` parameter.

# In[ ]:

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

# In[ ]:

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