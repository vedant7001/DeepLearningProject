"""
Training script for Question Answering models.

This module provides functions for training the QA models on SQuAD.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.squad_preprocessing import get_squad_dataloader
from models.lstm_model import LSTMEncoderDecoder, QAModel
from models.attn_model import LSTMEncoderAttnDecoder
from models.transformer_qa import TransformerQA, create_transformer_qa
from utils.metrics import compute_metrics
from utils.tokenization import get_tokenizer, decode_tokens
from utils.visualization import plot_training_progress, plot_loss_curve, save_attention_examples

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class SQuADDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get question and context
        question = item['question']
        context = item['context']
        
        # For training data, get the answer
        if 'answers' in item:
            answer_start = item['answers']['answer_start'][0]
            answer_text = item['answers']['text'][0]
        else:
            answer_start = 0
            answer_text = ""

        # Tokenize inputs
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension added by tokenizer
        inputs = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add answer information for training
        if answer_text:
            inputs['answer_start'] = torch.tensor(answer_start)
            inputs['answer_text'] = answer_text

        return inputs


def get_tokenizer(tokenizer_name: str):
    """Initialize the tokenizer"""
    return AutoTokenizer.from_pretrained(tokenizer_name)


def get_squad_dataloader(
    data_split: str,
    tokenizer_name: str,
    batch_size: int,
    shuffle: bool = True,
    max_length: int = 384,
    use_v2: bool = False,
    num_workers: int = 2
) -> DataLoader:
    """Create a DataLoader for SQuAD dataset"""
    
    # Load dataset
    dataset_name = "squad_v2" if use_v2 else "squad"
    dataset = load_dataset(dataset_name, split=data_split)
    
    # Initialize tokenizer and create dataset
    tokenizer = get_tokenizer(tokenizer_name)
    squad_dataset = SQuADDataset(dataset, tokenizer, max_length)
    
    # Create dataloader
    return DataLoader(
        squad_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    clip_grad_norm: float = 1.0
) -> Tuple[float, List[float]]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    batch_losses = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Extract and handle the start and end positions
        start_positions = batch.get('answer_start')
        
        # For QA tasks, we often need to calculate the end position based on the start position
        # and answer length. However, if end positions are already provided, use those.
        end_positions = batch.get('answer_end')
        if end_positions is None and start_positions is not None and 'answer_text' in batch:
            # For simplicity, we could approximate end positions by adding the length of answer text
            # to start position. This is a simplification as token lengths may differ.
            # In a real implementation, tokenize the answer text and get its length.
            end_positions = start_positions + 1  # Default fallback is 1 token after start
        
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            start_positions=start_positions,
            end_positions=end_positions
        )
        
        # Extract loss from outputs
        if isinstance(outputs, dict):
            loss = outputs.get('loss')
        else:
            # For compatibility with models that return loss directly
            loss = outputs
            
        if loss is None:
            # Fallback in case model doesn't compute loss internally
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Update progress
        total_loss += loss.item()
        batch_losses.append(loss.item())
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader), batch_losses


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get predictions
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Extract predictions from outputs
            if isinstance(outputs, dict):
                start_logits = outputs.get('start_logits')
                end_logits = outputs.get('end_logits')
                
                # Get the most likely start and end positions
                start_indices = torch.argmax(start_logits, dim=1)
                end_indices = torch.argmax(end_logits, dim=1)
                
                # Create prediction pairs
                for i in range(len(start_indices)):
                    start_idx = start_indices[i].item()
                    end_idx = end_indices[i].item()
                    
                    # Make sure end comes after start
                    if end_idx < start_idx:
                        end_idx = start_idx
                    
                    # Extract the predicted answer
                    answer_tokens = batch['input_ids'][i][start_idx:end_idx+1]
                    all_predictions.append((start_idx, end_idx))
            else:
                # For compatibility with models that might return predictions directly
                all_predictions.extend(outputs.get('predictions', []))
            
            # Store labels if available
            if 'answer_start' in batch and 'answer_text' in batch:
                for start_idx, answer_text in zip(batch['answer_start'], batch['answer_text']):
                    all_labels.append((start_idx.item(), answer_text))
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels) if all_labels else {}
    return metrics


def train(
    model_type: str = "lstm",
    tokenizer_name: str = "bert-base-uncased",
    output_dir: str = "Result/model_outputs",
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_length: int = 384,
    clip_grad_norm: float = 1.0,
    num_workers: int = 2,
    embed_size: int = 128,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    seed: int = 42,
    **kwargs  # Accept additional keyword arguments
) -> None:
    """Main training function"""
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = QAModel(
        model_type=model_type,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Get dataloaders
    use_v2 = kwargs.get('use_v2', False)
    train_dataloader = get_squad_dataloader(
        "train",
        tokenizer_name,
        train_batch_size,
        max_length=max_length,
        num_workers=num_workers,
        use_v2=use_v2
    )
    
    eval_dataloader = get_squad_dataloader(
        "validation",
        tokenizer_name,
        eval_batch_size,
        shuffle=False,
        max_length=max_length,
        num_workers=num_workers,
        use_v2=use_v2
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_metric = float('-inf')
    checkpoint_every = kwargs.get('checkpoint_every', 1)
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, batch_losses = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            clip_grad_norm
        )
        logger.info(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(model, eval_dataloader, device)
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Save best model
        current_metric = metrics.get('f1', float('-inf'))
        if current_metric > best_metric:
            best_metric = current_metric
            model_path = os.path.join(output_dir, f"{model_type}_model_best.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(output_dir, f"{model_type}_checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'metrics': metrics
            }, checkpoint_path)
        
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a QA model")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "attn", "transformer"],
                        help="Type of model to train")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                        help="Name of the Hugging Face tokenizer to use")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save model and logs")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of epochs to train")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--use_v2", action="store_true",
                        help="Whether to use SQuAD v2.0 instead of v1.1")
    parser.add_argument("--checkpoint_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log to tensorboard every N steps")
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"],
                        help="Size of transformer model")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5,
                        help="Probability of using teacher forcing (for LSTM models)")
    parser.add_argument("--embed_size", type=int, default=128,
                        help="Size of embeddings (for LSTM models)")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Size of hidden states (for LSTM models)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument("--save_attention", action="store_true",
                        help="Whether to save attention weights")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory with model type
    output_dir = os.path.join(args.output_dir, args.model_type)
    
    # Train model
    train(
        model_type=args.model_type,
        tokenizer_name=args.tokenizer_name,
        output_dir=output_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        clip_grad_norm=args.clip_grad_norm,
        use_v2=args.use_v2,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        model_size=args.model_size,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        save_attention=args.save_attention,
        seed=args.seed
    )
