import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Deep Learning Question Answering System\n\nThis notebook helps you run the QA system on Google Colab."
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Clone the repository\n!git clone https://github.com/vedant7001/DeepLearningProject.git\n%cd DeepLearningProject"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Install dependencies\n!pip install torch transformers numpy tqdm tensorboard pandas scikit-learn matplotlib"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Setup Python path properly\nimport sys\nimport os\n\n# Get current working directory\n!pwd\n\n# Add the current directory to the Python path\nsys.path.insert(0, os.getcwd())\n\n# Create empty __init__.py files to make directories importable\n!touch __init__.py\n!touch training/__init__.py\n!touch models/__init__.py\n!touch utils/__init__.py\n\n# Verify paths\nprint(f\"Python path: {sys.path}\")\nprint(f\"Directory contents: {os.listdir()}\")\nprint(f\"Training directory contents: {os.listdir('training')}\")\n\n# Verify GPU\nimport torch\nprint(f\"GPU available: {torch.cuda.is_available()}\")\nif torch.cuda.is_available():\n    print(f\"GPU device: {torch.cuda.get_device_name(0)}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Configuration"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Training configuration\nconfig = {\n    'model_type': 'lstm',  # Options: 'lstm', 'attn', 'transformer'\n    'batch_size': 16,\n    'num_epochs': 20,\n    'embedding_size': 128,\n    'hidden_size': 256,\n    'learning_rate': 3e-4,\n    'dropout': 0.3,\n    'num_layers': 2\n}"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Direct Import Test"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Test imports directly\ntry:\n    import training\n    print(\"✅ Successfully imported training module\")\n    print(f\"Training module path: {training.__file__}\")\n    \n    import training.train\n    print(\"✅ Successfully imported training.train module\")\n    print(f\"Training.train module path: {training.train.__file__}\")\n    \n    from training.train import train\n    print(\"✅ Successfully imported train function\")\n    \n    import models\n    print(\"✅ Successfully imported models module\")\n    \n    import utils\n    print(\"✅ Successfully imported utils module\")\nexcept Exception as e:\n    print(f\"❌ Import failed: {e}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Data Loading Test"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Test data loading functions\ntry:\n    from training.train import get_squad_dataloader, get_tokenizer\n    \n    # Test tokenizer loading\n    tokenizer_name = 'bert-base-uncased'\n    tokenizer = get_tokenizer(tokenizer_name)\n    print(\"✅ Successfully loaded tokenizer:\")\n    print(f\"Vocabulary size: {tokenizer.vocab_size}\")\n    print(f\"Padding token: {tokenizer.pad_token}\")\n    print(f\"Beginning token: {tokenizer.bos_token}\")\n    print(f\"End token: {tokenizer.eos_token}\")\n    \n    # Test dataloader creation\n    try:\n        print(\"\\nTrying to load SQuAD training data...\")\n        train_dataloader = get_squad_dataloader(\n            data_split=\"train\",\n            tokenizer_name=tokenizer_name,\n            batch_size=2,  # Small batch for testing\n            shuffle=False,\n            use_v2=False\n        )\n        print(f\"✅ Successfully created dataloader with {len(train_dataloader)} batches\")\n        \n        # Get a sample batch\n        batch = next(iter(train_dataloader))\n        print(\"\\nSample batch keys:\", batch.keys())\n        \n        # Show a sample question and context\n        sample_idx = 0\n        sample_question = batch['questions'][sample_idx]\n        sample_context = batch['contexts'][sample_idx]\n        print(f\"\\nSample question: {sample_question}\")\n        print(f\"Sample context (truncated): {sample_context[:100]}...\")\n        \n    except Exception as e:\n        print(f\"❌ Data loading failed: {e}\")\n        import traceback\n        traceback.print_exc()\n        print(\"\\nThis may be because the dataset files are not available.\")\n        print(\"Try downloading the SQuAD dataset or using a different dataset.\")\n\nexcept Exception as e:\n    print(f\"❌ Data loading test failed: {e}\")\n    import traceback\n    traceback.print_exc()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Training"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Import and run training\ntry:\n    from training.train import train\n\n    # Create output directory\n    output_dir = 'Result/model_outputs'\n    os.makedirs(output_dir, exist_ok=True)\n\n    # Start training\n    train(\n        model_type=config['model_type'],\n        tokenizer_name='bert-base-uncased',\n        output_dir=output_dir,\n        train_batch_size=config['batch_size'],\n        eval_batch_size=config['batch_size'],  # Using same batch size for eval\n        num_epochs=config['num_epochs'],\n        embed_size=config['embedding_size'],\n        hidden_size=config['hidden_size'],\n        learning_rate=config['learning_rate'],\n        dropout=config['dropout'],\n        num_layers=config['num_layers'],\n        weight_decay=0.01,\n        warmup_steps=1000,\n        checkpoint_every=1,\n        log_every=100,\n        teacher_forcing_ratio=0.5,\n        save_attention=True,\n        model_size=\"base\",  # For transformer models\n        seed=42\n    )\nexcept Exception as e:\n    print(f\"❌ Training failed: {e}\")\n    import traceback\n    traceback.print_exc()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Inference"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Run inference\ntry:\n    from training.predict import predict\n\n    # Load the best model\n    model_path = os.path.join(output_dir, f\"{config['model_type']}_model_best.pt\")\n\n    # Example usage\n    context = \"\"\"The quick brown fox jumps over the lazy dog. The dog was sleeping in the sun.\"\"\"\n    question = \"What did the fox do?\"\n\n    answer = predict(\n        model_path=model_path,\n        question=question,\n        context=context,\n        model_type=config['model_type']\n    )\n\n    print(f\"Question: {question}\")\n    print(f\"Answer: {answer}\")\nexcept Exception as e:\n    print(f\"❌ Inference failed: {e}\")"
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open('QA_System_Colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook created successfully!") 