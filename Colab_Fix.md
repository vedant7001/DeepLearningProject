# Fix for ModuleNotFoundError in Google Colab

When running the code in Google Colab, you may encounter the error `ModuleNotFoundError: No module named 'training'`. This is because Python is looking for modules in the wrong location. Here's how to fix it:

## Replace Step 4 with this improved version:

```python
# Set up the Python path correctly
import sys
import os

# Get the current working directory (should be inside the DeepLearningProject folder)
repo_path = os.path.abspath('.')
print(f"Current directory: {repo_path}")

# List all files and directories to verify what we're working with
print("\nFiles in current directory:")
!ls -la

# Add the current directory to Python path
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
    print(f"\nAdded {repo_path} to Python path")

# Create an empty __init__.py file in the training directory if it doesn't exist
training_init_path = os.path.join(repo_path, 'training', '__init__.py')
if not os.path.exists(training_init_path):
    !mkdir -p training
    !touch training/__init__.py
    print(f"Created {training_init_path}")

# Check if we can now import the training module
try:
    import training
    print("Successfully imported training module")
except ImportError as e:
    print(f"Error importing training module: {e}")

# Print the Python path for debugging
print("\nPython path:")
for p in sys.path:
    print(f"  {p}")
```

## If you still have issues, try this direct approach:

```python
# Alternative fix: Run the training script directly without import
!python -c "import sys, os; sys.path.insert(0, os.path.abspath('.')); from training.train import train; print('Train function imported successfully')"

# If the above works, you can run training directly with this approach:
!PYTHONPATH=$PYTHONPATH:/content/DeepLearningProject python Result/main.py train --model_type lstm --embed_size 128 --hidden_size 256 --num_layers 2 --dropout 0.3 --learning_rate 1e-3 --num_epochs 5 --train_batch_size 32 --eval_batch_size 32 --output_dir /content/drive/MyDrive/DeepLearningProject/model_outputs
```

## If the above doesn't work, you might need to check the structure:

```python
# Inspect the repository structure
!find . -type f -name "*.py" | sort
``` 