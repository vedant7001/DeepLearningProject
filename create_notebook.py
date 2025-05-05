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
            "source": "# Clone the repository\n!git clone https://github.com/vedant7001/DeepLearningProject.git\n!ls -la"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Navigate to the repository directory\n%cd DeepLearningProject\n!ls -la"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Install dependencies\n!pip install torch transformers numpy tqdm tensorboard pandas scikit-learn matplotlib datasets"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Setup Python path properly\nimport sys\nimport os\n\n# Get current working directory\n!pwd\n\n# Add the current directory to the Python path\nsys.path.insert(0, os.getcwd())\n\n# Check if directories exist\n!ls -la\nprint(\"\\nChecking for required directories:\")\ndirectories = ['training', 'models', 'utils', 'Result']\nfor directory in directories:\n    if os.path.exists(directory):\n        print(f\"✅ {directory} directory exists\")\n        print(f\"   Contents: {os.listdir(directory)}\")\n    else:\n        print(f\"❌ {directory} directory does not exist\")\n\n# Create empty __init__.py files if directories exist\nfor directory in directories:\n    if os.path.exists(directory):\n        init_file = os.path.join(directory, '__init__.py')\n        if not os.path.exists(init_file):\n            print(f\"Creating {init_file}\")\n            with open(init_file, 'w') as f:\n                f.write('# Auto-generated __init__.py for Python package')\n\n# Verify Python path\nprint(f\"\\nPython path: {sys.path}\")\n\n# Verify GPU\nimport torch\nprint(f\"\\nGPU available: {torch.cuda.is_available()}\")\nif torch.cuda.is_available():\n    print(f\"GPU device: {torch.cuda.get_device_name(0)}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Check for Missing Dependencies"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Check for essential dependencies and install if missing\ndef check_and_install_dependencies():\n    required_packages = {\n        'datasets': 'datasets',\n        'torch': 'torch',\n        'transformers': 'transformers',\n        'numpy': 'numpy',\n        'tqdm': 'tqdm',\n        'pandas': 'pandas',\n        'sklearn': 'scikit-learn',\n        'matplotlib': 'matplotlib',\n        'nltk': 'nltk'\n    }\n    \n    missing_packages = []\n    \n    for module_name, package_name in required_packages.items():\n        try:\n            __import__(module_name)\n            print(f\"✅ {module_name} is already installed\")\n        except ImportError:\n            print(f\"❌ {module_name} is missing\")\n            missing_packages.append(package_name)\n    \n    if missing_packages:\n        print(f\"\\nInstalling missing packages: {', '.join(missing_packages)}\")\n        for package in missing_packages:\n            !pip install {package}\n            print(f\"Installed {package}\")\n    else:\n        print(\"\\nAll required packages are installed!\")\n\n# Run the dependency checker\ncheck_and_install_dependencies()\n\n# Special handling for NLTK data (often needed for NLP tasks)\ntry:\n    import nltk\n    nltk.download('punkt')\n    nltk.download('stopwords')\n    print(\"✅ NLTK data downloaded successfully\")\nexcept Exception as e:\n    print(f\"❌ Error downloading NLTK data: {e}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Directory Structure Analysis"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Analyze repository structure\ndef list_dir_recursive(path, level=0):\n    if not os.path.exists(path):\n        print(f\"Path does not exist: {path}\")\n        return\n        \n    if os.path.isfile(path):\n        print(f\"{'  ' * level}📄 {os.path.basename(path)}\")\n        return\n        \n    print(f\"{'  ' * level}📁 {os.path.basename(path) or path}\")\n    try:\n        for item in sorted(os.listdir(path)):\n            list_dir_recursive(os.path.join(path, item), level + 1)\n    except PermissionError:\n        print(f\"{'  ' * (level+1)}⛔ Permission denied\")\n\nprint(\"Complete repository structure:\")\nlist_dir_recursive('.')"
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
            "source": "# Test imports directly\ntry:\n    import training\n    print(\"✅ Successfully imported training module\")\n    print(f\"Training module path: {training.__file__}\")\n    \n    import training.train\n    print(\"✅ Successfully imported training.train module\")\n    print(f\"Training.train module path: {training.train.__file__}\")\n    \n    from training.train import train\n    print(\"✅ Successfully imported train function\")\n    \n    import models\n    print(\"✅ Successfully imported models module\")\n    \n    import utils\n    print(\"✅ Successfully imported utils module\")\nexcept Exception as e:\n    print(f\"❌ Import failed: {e}\")\n    import traceback\n    traceback.print_exc()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Manual Directory Creation (If Needed)"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Only run this cell if directories are missing\ndef create_missing_directories():\n    # Create missing directories and basic files\n    for directory in ['training', 'models', 'utils', 'Result']:\n        if not os.path.exists(directory):\n            print(f\"Creating directory: {directory}\")\n            os.makedirs(directory, exist_ok=True)\n            \n            # Create __init__.py\n            with open(os.path.join(directory, '__init__.py'), 'w') as f:\n                f.write(f'# Auto-generated __init__.py for {directory} package\\n')\n    \n    # Create empty placeholder files if training.py doesn't exist\n    if not os.path.exists('training/train.py'):\n        print(\"Creating basic train.py placeholder\")\n        with open('training/train.py', 'w') as f:\n            f.write(\"\"\"\n# Placeholder train.py file\ndef train(model_type, tokenizer_name, output_dir, **kwargs):\n    print(f\"Training with {model_type} model\")\n    print(f\"Using tokenizer: {tokenizer_name}\")\n    print(f\"Output directory: {output_dir}\")\n    print(f\"Additional parameters: {kwargs}\")\n    print(\"This is a placeholder function. Please upload actual training code.\")\n    return None\n\"\"\")\n            \n# Only uncomment and run this code if directories are missing\n# create_missing_directories()"
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
            "source": "# Run inference\ntry:\n    from training.predict import predict\n\n    # Load the best model\n    model_path = os.path.join(output_dir, f\"{config['model_type']}_model_best.pt\")\n\n    # Example usage\n    context = \"\"\"The quick brown fox jumps over the lazy dog. The dog was sleeping in the sun.\"\"\"\n    question = \"What did the fox do?\"\n\n    answer = predict(\n        model_path=model_path,\n        question=question,\n        context=context,\n        model_type=config['model_type']\n    )\n\n    print(f\"Question: {question}\")\n    print(f\"Answer: {answer}\")\nexcept Exception as e:\n    print(f\"❌ Inference failed: {e}\")\n    import traceback\n    traceback.print_exc()"
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