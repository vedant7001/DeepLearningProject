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
            "source": "# Setup Python path and check GPU\nimport sys\nimport os\nsys.path.append(os.getcwd())\n\n# Verify GPU\nimport torch\nprint(f\"GPU available: {torch.cuda.is_available()}\")\nif torch.cuda.is_available():\n    print(f\"GPU device: {torch.cuda.get_device_name(0)}\")"
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
            "source": "# Import and run training\nfrom training.train import train\n\n# Create output directory\noutput_dir = 'Result/model_outputs'\nos.makedirs(output_dir, exist_ok=True)\n\n# Start training\ntrain(\n    model_type=config['model_type'],\n    tokenizer_name='bert-base-uncased',\n    output_dir=output_dir,\n    train_batch_size=config['batch_size'],\n    num_epochs=config['num_epochs'],\n    embed_size=config['embedding_size'],\n    hidden_size=config['hidden_size'],\n    learning_rate=config['learning_rate'],\n    dropout=config['dropout'],\n    num_layers=config['num_layers']\n)"
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
            "source": "# Run inference\nfrom training.predict import predict\n\n# Load the best model\nmodel_path = os.path.join(output_dir, f\"{config['model_type']}_model_best.pt\")\n\n# Example usage\ncontext = \"\"\"The quick brown fox jumps over the lazy dog. The dog was sleeping in the sun.\"\"\"\nquestion = \"What did the fox do?\"\n\nanswer = predict(\n    model_path=model_path,\n    question=question,\n    context=context,\n    model_type=config['model_type']\n)\n\nprint(f\"Question: {question}\")\nprint(f\"Answer: {answer}\")"
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