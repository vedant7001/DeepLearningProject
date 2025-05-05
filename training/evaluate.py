"""
Evaluation script for Question Answering models.

This module provides functions for evaluating QA models on SQuAD.
"""

import os
import sys
import json
import logging
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.squad_preprocessing import get_squad_dataloader
from models.lstm_model import LSTMEncoderDecoder
from models.attn_model import LSTMEncoderAttnDecoder
from models.transformer_qa import TransformerQA, create_transformer_qa
from utils.metrics import compute_metrics, compute_f1, compute_exact_match, compute_bleu, compute_rouge
from utils.tokenization import get_tokenizer
from utils.visualization import plot_model_comparison, save_attention_examples

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_model(
    model_type: str,
    model_path: str,
    vocab_size: int,
    device: torch.device,
    model_size: str = "base",
    embed_size: int = 256,
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.3,
    padding_idx: int = 0,
    sos_token_id: int = 1,
    eos_token_id: int = 2,
    save_attention: bool = True
) -> torch.nn.Module:
    """
    Load a QA model from a checkpoint.
    
    Args:
        model_type: Type of model ("lstm", "attn", "transformer").
        model_path: Path to model checkpoint.
        vocab_size: Size of vocabulary.
        device: Device to load model on.
        model_size: Size of transformer model ("small", "base", "large").
        embed_size: Size of embeddings (for LSTM models).
        hidden_size: Size of hidden states.
        num_layers: Number of layers.
        dropout: Dropout probability.
        padding_idx: Padding token ID.
        sos_token_id: Start of sequence token ID.
        eos_token_id: End of sequence token ID.
        save_attention: Whether to save attention weights.
    
    Returns:
        Loaded model.
    """
    # Create a new model
    if model_type == "lstm":
        model = LSTMEncoderDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id
        )
    elif model_type == "attn":
        model = LSTMEncoderAttnDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            save_attention=save_attention
        )
    elif model_type == "transformer":
        model = create_transformer_qa(
            vocab_size=vocab_size,
            model_size=model_size,
            padding_idx=padding_idx,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            dropout=dropout,
            save_attention=save_attention
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model = model.to(device)
    
    return model


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer,
    device: torch.device,
    output_dir: str,
    model_name: str,
    max_eval_samples: int = -1,
    save_results: bool = True,
    save_attention: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a model on SQuAD.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader for evaluation data.
        tokenizer: Tokenizer for decoding predictions.
        device: Device to evaluate on.
        output_dir: Directory to save results.
        model_name: Name of the model.
        max_eval_samples: Maximum number of samples to evaluate (-1 for all).
        save_results: Whether to save results to disk.
        save_attention: Whether to save attention visualizations.
        
    Returns:
        Dictionary with evaluation results.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_contexts = []
    all_questions = []
    all_attention_weights = []
    all_tokens = []
    all_example_ids = []
    
    # For attention visualization (if supported by the model)
    has_attention = (
        (hasattr(model, "save_attention") and model.save_attention) or
        (isinstance(model, LSTMEncoderAttnDecoder) and hasattr(model, "get_attention_weights"))
    )
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"Evaluating {model_name}")
    
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # Check if we've reached the maximum number of samples
            if max_eval_samples > 0 and i * batch["encoder_inputs"].size(0) >= max_eval_samples:
                break
            
            # Move batch to device
            encoder_inputs = batch["encoder_inputs"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            # Generate predictions
            generation_outputs = model.generate(
                encoder_inputs=encoder_inputs,
                encoder_mask=encoder_mask,
                max_length=50
            )
            
            # Get generated IDs
            generated_ids = generation_outputs["generated_ids"]
            
            # Get attention weights if available (for visualization)
            if has_attention:
                if "attention_weights" in generation_outputs:
                    batch_attention = generation_outputs["attention_weights"].cpu().numpy()
                elif hasattr(model, "get_attention_weights"):
                    attention = model.get_attention_weights()
                    if attention is not None:
                        batch_attention = attention.cpu().numpy()
                    else:
                        batch_attention = None
                else:
                    batch_attention = None
                
                if batch_attention is not None:
                    all_attention_weights.extend(batch_attention)
                    
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
                
                # Store context, question, and target
                all_contexts.append(batch["contexts"][j])
                all_questions.append(batch["questions"][j])
                all_targets.append(batch["answers"][j])
                all_example_ids.append(batch["ids"][j])
    
    # Compute metrics
    metrics = compute_metrics(
        predictions=all_predictions,
        ground_truths=all_targets,
        include_rouge=True,
        include_bleu=True
    )
    
    # Create examples for inspection
    num_examples = min(10, len(all_predictions))
    examples = []
    for i in range(num_examples):
        example = {
            "id": all_example_ids[i],
            "context": all_contexts[i],
            "question": all_questions[i],
            "prediction": all_predictions[i],
            "answer": all_targets[i],
            "f1": compute_f1(all_predictions[i], all_targets[i]) * 100,
            "exact_match": compute_exact_match(all_predictions[i], all_targets[i]) * 100
        }
        examples.append(example)
    
    # Create evaluation results
    results = {
        "model_name": model_name,
        "metrics": metrics,
        "examples": examples,
        "predictions": all_predictions,
        "targets": all_targets,
        "contexts": all_contexts,
        "questions": all_questions,
        "ids": all_example_ids
    }
    
    # Print results
    logger.info(f"Evaluation results for {model_name}:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.2f}")
    
    # Print example predictions
    logger.info("\nExample predictions:")
    for i, example in enumerate(examples):
        logger.info(f"Example {i + 1} (ID: {example['id']}):")
        logger.info(f"  Context (truncated): {example['context'][:100]}...")
        logger.info(f"  Question: {example['question']}")
        logger.info(f"  Prediction: {example['prediction']}")
        logger.info(f"  Answer: {example['answer']}")
        logger.info(f"  F1: {example['f1']:.2f}, Exact Match: {example['exact_match']:.2f}")
        logger.info("")
    
    # Save results if requested
    if save_results:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete results
        results_path = os.path.join(output_dir, f"{model_name}_results.json")
        with open(results_path, "w") as f:
            # Convert numpy values to Python native types
            clean_metrics = {k: float(v) for k, v in metrics.items()}
            json.dump({
                "model_name": model_name,
                "metrics": clean_metrics,
                "examples": examples,
                "predictions": all_predictions,
                "targets": all_targets,
                "ids": all_example_ids
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Save attention visualizations if available
        if save_attention and all_attention_weights and all_tokens:
            save_attention_examples(
                examples=[{
                    "context": all_contexts[i],
                    "question": all_questions[i],
                    "prediction": all_predictions[i],
                    "answer": all_targets[i]
                } for i in range(min(5, len(all_predictions)))],
                attentions=all_attention_weights[:5],
                tokens_list=all_tokens[:5],
                model_name=model_name,
                output_dir=os.path.join(output_dir, "attention_visualizations"),
                num_examples=5
            )
    
    return results


def compare_models(model_results: List[Dict[str, Any]], output_dir: str):
    """
    Compare results from multiple models.
    
    Args:
        model_results: List of results dictionaries from multiple models.
        output_dir: Directory to save comparison results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for comparison
    model_names = [results["model_name"] for results in model_results]
    metrics_to_compare = ["exact_match", "f1", "bleu", "rougeL_f"]
    metrics_dict = {metric: [] for metric in metrics_to_compare}
    
    for results in model_results:
        for metric in metrics_to_compare:
            if metric in results["metrics"]:
                metrics_dict[metric].append(results["metrics"][metric])
            else:
                metrics_dict[metric].append(0.0)
    
    # Create a table for side-by-side comparison
    table_data = []
    header = ["Model"] + [metric.upper() for metric in metrics_to_compare]
    for i, model_name in enumerate(model_names):
        row = [model_name]
        for metric in metrics_to_compare:
            if i < len(metrics_dict[metric]):
                row.append(f"{metrics_dict[metric][i]:.2f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Print the table
    logger.info("Model Comparison:")
    logger.info(tabulate(table_data, headers=header, tablefmt="grid"))
    
    # Save comparison to file
    comparison_path = os.path.join(output_dir, "model_comparison.txt")
    with open(comparison_path, "w") as f:
        f.write(tabulate(table_data, headers=header, tablefmt="grid"))
    logger.info(f"Model comparison saved to {comparison_path}")
    
    # Create visualization
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plot_model_comparison(
        model_names=model_names,
        metrics=metrics_dict,
        save_path=plot_path,
        title="Model Performance Comparison"
    )
    logger.info(f"Model comparison plot saved to {plot_path}")


def evaluate_all_models(
    model_configs: List[Dict[str, Any]],
    tokenizer_name: str,
    data_split: str,
    batch_size: int,
    output_dir: str,
    max_eval_samples: int = -1,
    save_results: bool = True,
    use_v2: bool = False,
    save_attention: bool = True
):
    """
    Evaluate multiple models on SQuAD.
    
    Args:
        model_configs: List of model configuration dictionaries.
        tokenizer_name: Name of the Hugging Face tokenizer to use.
        data_split: Which split to evaluate on ("train", "val", "test").
        batch_size: Batch size for evaluation.
        output_dir: Directory to save results.
        max_eval_samples: Maximum number of samples to evaluate (-1 for all).
        save_results: Whether to save results to disk.
        use_v2: Whether to use SQuAD v2.0 instead of v1.1.
        save_attention: Whether to save attention visualizations.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = get_tokenizer(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    sos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    # Create dataloader
    dataloader = get_squad_dataloader(
        data_split=data_split,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        shuffle=False,
        use_v2=use_v2
    )
    
    # Evaluate each model
    all_results = []
    for config in model_configs:
        # Load model
        model = load_model(
            model_type=config["model_type"],
            model_path=config["model_path"],
            vocab_size=vocab_size,
            device=device,
            model_size=config.get("model_size", "base"),
            embed_size=config.get("embed_size", 256),
            hidden_size=config.get("hidden_size", 512),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.3),
            padding_idx=pad_token_id,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            save_attention=save_attention
        )
        
        # Evaluate model
        model_name = config.get("name", f"{config['model_type']}")
        model_output_dir = os.path.join(output_dir, model_name)
        
        results = evaluate_model(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            device=device,
            output_dir=model_output_dir,
            model_name=model_name,
            max_eval_samples=max_eval_samples,
            save_results=save_results,
            save_attention=save_attention
        )
        
        all_results.append(results)
    
    # Compare models
    if len(all_results) > 1:
        compare_models(all_results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA models")
    parser.add_argument("--model_configs", type=str, required=True,
                        help="Path to JSON file with model configurations")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                        help="Name of the Hugging Face tokenizer to use")
    parser.add_argument("--data_split", type=str, default="val", choices=["train", "val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--max_eval_samples", type=int, default=-1,
                        help="Maximum number of samples to evaluate (-1 for all)")
    parser.add_argument("--save_results", action="store_true",
                        help="Whether to save results to disk")
    parser.add_argument("--use_v2", action="store_true",
                        help="Whether to use SQuAD v2.0 instead of v1.1")
    parser.add_argument("--save_attention", action="store_true",
                        help="Whether to save attention visualizations")
    
    args = parser.parse_args()
    
    # Load model configurations
    with open(args.model_configs, "r") as f:
        model_configs = json.load(f)
    
    # Evaluate models
    evaluate_all_models(
        model_configs=model_configs,
        tokenizer_name=args.tokenizer_name,
        data_split=args.data_split,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        max_eval_samples=args.max_eval_samples,
        save_results=args.save_results,
        use_v2=args.use_v2,
        save_attention=args.save_attention
    )
