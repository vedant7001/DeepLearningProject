# Deep Learning Question Answering System

A PyTorch-based Question Answering system that uses advanced deep learning architectures including LSTM, Attention mechanisms, and Transformers.

## Project Structure
```
qa_system/
├── data/           # Dataset and preprocessing scripts
├── models/         # Neural network architectures
│   ├── lstm_model.py
│   ├── attn_model.py
│   └── transformer_qa.py
├── training/       # Training and evaluation
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── utils/          # Utility functions
├── Result/         # Output and checkpoints
└── QA_System_Colab.ipynb
```

## Features

- Multiple model architectures (LSTM, Attention, Transformer)
- Efficient data processing pipeline
- Support for both local and Google Colab execution
- Comprehensive evaluation metrics
- Experiment tracking with TensorBoard/W&B

## Setup Instructions

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/qa-system.git
cd qa-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Google Colab Setup

1. Open `QA_System_Colab.ipynb` in Google Colab
2. Follow the setup instructions in the notebook
3. Make sure to select GPU runtime for better performance

## Model Configuration

Default hyperparameters:
- Batch size: 16
- Number of epochs: 20
- Embedding size: 128
- Hidden size: 256
- Learning rate: 3e-4

## Training

1. Prepare your dataset:
```bash
python -m utils.prepare_data
```

2. Start training:
```bash
python -m training.train --model lstm --batch-size 16
```

3. Monitor training:
```bash
tensorboard --logdir Result/logs
```

## Evaluation

Run evaluation on test set:
```bash
python -m training.evaluate --model-path Result/checkpoints/best_model.pth
```

## Inference

Make predictions on new questions:
```bash
python -m training.predict --input "Your question here?"
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- HuggingFace team for transformer implementations
- The open-source community for various contributions

## Contact

Your Name - your.email@example.com
Project Link: [https://github.com/your-username/qa-system](https://github.com/your-username/qa-system) 