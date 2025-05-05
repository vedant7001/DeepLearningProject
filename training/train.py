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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.squad_preprocessing import get_squad_dataloader
from models.lstm_model import LSTMEncoderDecoder
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


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad_norm: float = 1.0
) -> Tuple[float, List[float]]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train.
        dataloader: DataLoader for training data.
        optimizer: Optimizer for updating weights.
        device: Device to train on.
        clip_grad_norm: Maximum gradient norm for gradient clipping.
    
    Returns:
        Tuple of average loss and list of all batch losses.
    """
    model.train()
    epoch_loss = 0.0
    batch_losses = []
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        encoder_inputs = batch["encoder_inputs"].to(device)
        decoder_inputs = batch["decoder_inputs"].to(device)
        targets = batch["targets"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        
        # Forward pass
        outputs = model(
            encoder_inputs=encoder_inputs,
            decoder_inputs=decoder_inputs,
            encoder_mask=encoder_mask,
            decoder_mask=decoder_mask,
            targets=targets
        )
        
        loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        # Update weights
        optimizer.step()
        
        # Update loss
        batch_loss = loss.item()
        epoch_loss += batch_loss
        batch_losses.append(batch_loss)
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
    
    # Calculate average loss
    avg_loss = epoch_loss / len(dataloader)
    
    return avg_loss, batch_losses


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    tokenizer,
    max_eval_samples: int = -1,
    compute_metrics_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader for validation data.
        device: Device to evaluate on.
        tokenizer: Tokenizer for decoding predictions.
        max_eval_samples: Maximum number of samples to evaluate (-1 for all).
        compute_metrics_fn: Function to compute metrics.
    
    Returns:
        Dictionary with evaluation results.
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_contexts = []
    all_questions = []
    all_attention_weights = []
    all_tokens = []
    
    # For attention visualization (if supported by the model)
    use_attention = hasattr(model, "save_attention") and model.save_attention
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # Check if we've reached the maximum number of samples
            if max_eval_samples > 0 and i * batch["encoder_inputs"].size(0) >= max_eval_samples:
                break
            
            # Move batch to device
            encoder_inputs = batch["encoder_inputs"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            # For loss calculation
            if "decoder_inputs" in batch and "targets" in batch:
                decoder_inputs = batch["decoder_inputs"].to(device)
                targets = batch["targets"].to(device)
                decoder_mask = batch["decoder_mask"].to(device)
                
                # Forward pass
                outputs = model(
                    encoder_inputs=encoder_inputs,
                    decoder_inputs=decoder_inputs,
                    encoder_mask=encoder_mask,
                    decoder_mask=decoder_mask,
                    targets=targets
                )
                
                loss = outputs["loss"]
                total_loss += loss.item()
            
            # Generate predictions
            generation_outputs = model.generate(
                encoder_inputs=encoder_inputs,
                encoder_mask=encoder_mask,
                max_length=50
            )
            
            # Get generated IDs
            generated_ids = generation_outputs["generated_ids"]
            
            # Get attention weights if available (for visualization)
            if use_attention and "attention_weights" in generation_outputs:
                all_attention_weights.extend(
                    generation_outputs["attention_weights"].cpu().numpy()
                )
                
                # Store tokens for visualization (first 5 examples only)
                if len(all_tokens) < 5:
                    for j in range(min(5, encoder_inputs.size(0))):
                        # Get input tokens for visualization
                        tokens = [tokenizer.decode([token]) for token in encoder_inputs[j].cpu().tolist()]
                        all_tokens.append(tokens)
            
            # Decode predictions and targets
            for j in range(generated_ids.size(0)):
                pred_ids = generated_ids[j].tolist()
                
                # Find EOS token if present
                if model.eos_token_id in pred_ids:
                    pred_ids = pred_ids[:pred_ids.index(model.eos_token_id)]
                
                # Decode prediction
                prediction = tokenizer.decode(pred_ids, skip_special_tokens=True)
                all_predictions.append(prediction)
                
                # Store context and question
                all_contexts.append(batch["contexts"][j])
                all_questions.append(batch["questions"][j])
                
                # Store target (ground truth) if available
                if "targets" in batch and "answers" in batch:
                    target = batch["answers"][j]
                    all_targets.append(target)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader) if "targets" in batch else 0.0
    
    # Compute metrics if targets are available
    results = {"loss": avg_loss, "predictions": all_predictions}
    
    if all_targets and compute_metrics_fn:
        metrics = compute_metrics_fn(all_predictions, all_targets)
        results.update(metrics)
    
    # Add examples for inspection
    num_examples = min(5, len(all_predictions))
    examples = []
    for i in range(num_examples):
        example = {
            "context": all_contexts[i],
            "question": all_questions[i],
            "prediction": all_predictions[i],
            "answer": all_targets[i] if all_targets else "N/A"
        }
        examples.append(example)
    
    results["examples"] = examples
    
    # Add attention weights for visualization (if available)
    if all_attention_weights and all_tokens:
        results["attention_weights"] = all_attention_weights[:num_examples]
        results["tokens"] = all_tokens[:num_examples]
    
    return results


def train(
    model_type: str,
    tokenizer_name: str,
    output_dir: str,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    num_epochs: int = 10,
    warmup_steps: int = 1000,
    clip_grad_norm: float = 1.0,
    use_v2: bool = False,
    checkpoint_every: int = 1,
    log_every: int = 100,
    model_size: str = "base",
    teacher_forcing_ratio: float = 0.5,
    embed_size: int = 256,
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.3,
    save_attention: bool = True,
    seed: int = 42
):
    """
    Train a QA model.
    
    Args:
        model_type: Type of model to train ("lstm", "attn", "transformer").
        tokenizer_name: Name of the Hugging Face tokenizer to use.
        output_dir: Directory to save model and logs.
        train_batch_size: Batch size for training.
        eval_batch_size: Batch size for evaluation.
        learning_rate: Learning rate.
        weight_decay: Weight decay for regularization.
        num_epochs: Number of epochs to train.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        clip_grad_norm: Maximum gradient norm for gradient clipping.
        use_v2: Whether to use SQuAD v2.0 instead of v1.1.
        checkpoint_every: Save checkpoint every N epochs.
        log_every: Log to tensorboard every N steps.
        model_size: Size of transformer model ("small", "base", "large").
        teacher_forcing_ratio: Probability of using teacher forcing (for LSTM models).
        embed_size: Size of embeddings (for LSTM models).
        hidden_size: Size of hidden states (for LSTM models).
        num_layers: Number of layers.
        dropout: Dropout probability.
        save_attention: Whether to save attention weights.
        seed: Random seed.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger.info(f"Training {model_type} model")
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = get_tokenizer(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    sos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    # Create dataloaders
    train_dataloader = get_squad_dataloader(
        data_split="train",
        tokenizer_name=tokenizer_name,
        batch_size=train_batch_size,
        shuffle=True,
        use_v2=use_v2
    )
    
    val_dataloader = get_squad_dataloader(
        data_split="val",
        tokenizer_name=tokenizer_name,
        batch_size=eval_batch_size,
        shuffle=False,
        use_v2=use_v2
    )
    
    # Create model
    if model_type == "lstm":
        model = LSTMEncoderDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=pad_token_id,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        model_name = "lstm_encoder_decoder"
    elif model_type == "attn":
        model = LSTMEncoderAttnDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=pad_token_id,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            teacher_forcing_ratio=teacher_forcing_ratio,
            save_attention=save_attention
        )
        model_name = "lstm_encoder_attn_decoder"
    elif model_type == "transformer":
        model = create_transformer_qa(
            vocab_size=vocab_size,
            model_size=model_size,
            padding_idx=pad_token_id,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            dropout=dropout,
            save_attention=save_attention
        )
        model_name = f"transformer_{model_size}"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device
    model = model.to(device)
    
    # Count and log number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )
    
    # Initialize training state
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_metrics = {}
    
    # Train model
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, batch_losses = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            clip_grad_norm=clip_grad_norm
        )
        train_losses.append(train_loss)
        
        # Log batch losses to TensorBoard
        for i, loss in enumerate(batch_losses):
            step = epoch * len(train_dataloader) + i
            if i % log_every == 0:
                writer.add_scalar("train/batch_loss", loss, step)
        
        # Log epoch loss to TensorBoard
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        val_results = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            tokenizer=tokenizer,
            compute_metrics_fn=compute_metrics
        )
        val_loss = val_results["loss"]
        val_losses.append(val_loss)
        
        # Log validation metrics to TensorBoard
        writer.add_scalar("val/loss", val_loss, epoch)
        for metric_name, metric_value in val_results.items():
            if isinstance(metric_value, (int, float)) and metric_name != "loss":
                writer.add_scalar(f"val/{metric_name}", metric_value, epoch)
                
                # Store for plotting
                if metric_name not in val_metrics:
                    val_metrics[metric_name] = []
                val_metrics[metric_name].append(metric_value)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print example predictions
        logger.info("Example predictions:")
        for i, example in enumerate(val_results["examples"]):
            logger.info(f"Example {i + 1}:")
            logger.info(f"  Context: {example['context'][:100]}...")
            logger.info(f"  Question: {example['question']}")
            logger.info(f"  Prediction: {example['prediction']}")
            logger.info(f"  Answer: {example['answer']}")
        
        # Save attention visualizations if available
        if "attention_weights" in val_results and save_attention:
            save_attention_examples(
                examples=val_results["examples"],
                attentions=val_results["attention_weights"],
                tokens_list=val_results["tokens"],
                model_name=model_name,
                output_dir=os.path.join(output_dir, "plots"),
                num_examples=5
            )
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, f"{model_name}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "val_metrics": val_results
            }, best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(output_dir, f"{model_name}_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "val_metrics": val_results
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.pt")
    torch.save({
        "epoch": num_epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_metrics": val_results
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Plot training progress
    metrics_dict = {}
    for metric_name, metric_values in val_metrics.items():
        metrics_dict[metric_name] = metric_values
    
    plot_path = os.path.join(output_dir, "plots", f"{model_name}_training_progress.png")
    plot_training_progress(
        train_losses=train_losses,
        val_losses=val_losses,
        metrics=metrics_dict,
        save_path=plot_path,
        title=f"{model_name.replace('_', ' ').title()} Training Progress"
    )
    logger.info(f"Training progress plot saved to {plot_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a QA model")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "attn", "transformer"],
                        help="Type of model to train")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                        help="Name of the Hugging Face tokenizer to use")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save model and logs")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--num_epochs", type=int, default=10,
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
    parser.add_argument("--embed_size", type=int, default=256,
                        help="Size of embeddings (for LSTM models)")
    parser.add_argument("--hidden_size", type=int, default=512,
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
