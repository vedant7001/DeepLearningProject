"""
Prediction script for Question Answering models.

This module provides functions for making predictions with QA models.
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
from utils.metrics import compute_metrics, compute_f1, compute_exact_match
from utils.tokenization import get_tokenizer, encode_qa_pair
from utils.visualization import visualize_attention, plot_attention_for_sample

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
    model.eval()
    
    return model


def predict_single_example(
    model: torch.nn.Module,
    tokenizer,
    context: str,
    question: str,
    device: torch.device,
    max_length: int = 50,
    visualize: bool = False
) -> Dict[str, Any]:
    """
    Generate prediction for a single example.
    
    Args:
        model: Trained QA model.
        tokenizer: Tokenizer for encoding inputs and decoding outputs.
        context: Context text.
        question: Question text.
        device: Device to run the model on.
        max_length: Maximum length of generated answer.
        visualize: Whether to return attention visualizations.
        
    Returns:
        Dictionary with prediction and optionally attention weights.
    """
    # Encode the input
    encoded = encode_qa_pair(
        context=context,
        question=question,
        tokenizer=tokenizer,
        return_tensors=True
    )
    
    # Move tensors to device
    encoder_input = encoded["encoder_input"].unsqueeze(0).to(device)
    
    # Create attention mask (1 for tokens, 0 for padding)
    encoder_mask = torch.ones_like(encoder_input, dtype=torch.long, device=device)
    
    # Generate prediction
    with torch.no_grad():
        generation_output = model.generate(
            encoder_inputs=encoder_input,
            encoder_mask=encoder_mask,
            max_length=max_length
        )
    
    # Get generated IDs
    generated_ids = generation_output["generated_ids"][0].cpu().tolist()
    
    # Find EOS token if present
    if model.eos_token_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(model.eos_token_id)]
    
    # Decode prediction
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Create result
    result = {
        "context": context,
        "question": question,
        "prediction": prediction
    }
    
    # Include attention visualization if requested and available
    if visualize:
        # Check if the model has attention weights available
        has_attention = (
            hasattr(model, "save_attention") and model.save_attention and
            (
                "attention_weights" in generation_output or
                (hasattr(model, "get_attention_weights") and model.get_attention_weights() is not None)
            )
        )
        
        if has_attention:
            # Get attention weights
            if "attention_weights" in generation_output:
                attention_weights = generation_output["attention_weights"][0].cpu().numpy()
            elif hasattr(model, "get_attention_weights"):
                attention_weights = model.get_attention_weights()[0].cpu().numpy()
            else:
                attention_weights = None
            
            if attention_weights is not None:
                # Get tokens for visualization
                tokens = [tokenizer.decode([token]) for token in encoder_input[0].cpu().tolist()]
                result["attention_weights"] = attention_weights
                result["tokens"] = tokens
    
    return result


def batch_predict(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    device: torch.device,
    batch_size: int = 16,
    max_examples: int = -1,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate predictions for a dataset.
    
    Args:
        model: Trained QA model.
        tokenizer: Tokenizer for encoding inputs and decoding outputs.
        dataset: Dataset to predict on.
        device: Device to run the model on.
        batch_size: Batch size for predictions.
        max_examples: Maximum number of examples to predict (-1 for all).
        output_file: Path to save predictions (optional).
        
    Returns:
        List of prediction dictionaries.
    """
    model.eval()
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    all_predictions = []
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc="Predicting")
    
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # Check if we've reached the maximum number of examples
            if max_examples > 0 and i * batch_size >= max_examples:
                break
            
            # Move batch to device
            encoder_inputs = batch["encoder_inputs"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            # Generate predictions
            generation_outputs = model.generate(
                encoder_inputs=encoder_inputs,
                encoder_mask=encoder_mask
            )
            
            # Get generated IDs
            generated_ids = generation_outputs["generated_ids"]
            
            # Process each example in the batch
            for j in range(generated_ids.size(0)):
                # Get IDs for this example
                pred_ids = generated_ids[j].tolist()
                
                # Find EOS token if present
                if model.eos_token_id in pred_ids:
                    pred_ids = pred_ids[:pred_ids.index(model.eos_token_id)]
                
                # Decode prediction
                prediction = tokenizer.decode(pred_ids, skip_special_tokens=True)
                
                # Create prediction dictionary
                pred_dict = {
                    "id": batch["ids"][j] if "ids" in batch else f"example_{i*batch_size + j}",
                    "context": batch["contexts"][j],
                    "question": batch["questions"][j],
                    "prediction": prediction
                }
                
                # Add ground truth if available
                if "answers" in batch:
                    pred_dict["answer"] = batch["answers"][j]
                    pred_dict["exact_match"] = compute_exact_match(prediction, batch["answers"][j])
                    pred_dict["f1"] = compute_f1(prediction, batch["answers"][j])
                
                all_predictions.append(pred_dict)
    
    # If an output file is specified, save predictions
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(all_predictions, f, indent=2)
    
    return all_predictions


def predict_from_file(
    model: torch.nn.Module,
    tokenizer,
    input_file: str,
    output_file: str,
    device: torch.device,
    visualize_attention: bool = False,
    visualization_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate predictions from a file of examples.
    
    Args:
        model: Trained QA model.
        tokenizer: Tokenizer for encoding inputs and decoding outputs.
        input_file: Path to input file (JSON format).
        output_file: Path to save predictions.
        device: Device to run the model on.
        visualize_attention: Whether to visualize attention weights.
        visualization_dir: Directory to save attention visualizations.
        
    Returns:
        List of prediction dictionaries.
    """
    # Load examples from file
    with open(input_file, "r") as f:
        examples = json.load(f)
    
    # Process each example
    all_predictions = []
    
    for i, example in enumerate(tqdm(examples, desc="Predicting")):
        # Extract context and question
        context = example["context"]
        question = example["question"]
        
        # Generate prediction
        prediction = predict_single_example(
            model=model,
            tokenizer=tokenizer,
            context=context,
            question=question,
            device=device,
            visualize=visualize_attention
        )
        
        # Add example ID if available
        if "id" in example:
            prediction["id"] = example["id"]
        else:
            prediction["id"] = f"example_{i}"
        
        # Add ground truth if available
        if "answer" in example:
            prediction["answer"] = example["answer"]
            prediction["exact_match"] = compute_exact_match(prediction["prediction"], example["answer"])
            prediction["f1"] = compute_f1(prediction["prediction"], example["answer"])
        
        # Visualize attention if requested
        if visualize_attention and visualization_dir and "attention_weights" in prediction:
            os.makedirs(visualization_dir, exist_ok=True)
            
            # Create visualization filename
            vis_filename = os.path.join(visualization_dir, f"attention_{prediction['id']}.png")
            
            # Create attention visualization
            plot_attention_for_sample(
                context=context,
                question=question,
                prediction=prediction["prediction"],
                ground_truth=example.get("answer", "N/A"),
                tokens=prediction["tokens"],
                attention_weights=prediction["attention_weights"],
                save_path=vis_filename,
                title=f"Attention Visualization - Example {prediction['id']}"
            )
            
            # Remove large arrays from the prediction dictionary
            del prediction["attention_weights"]
            del prediction["tokens"]
        
        all_predictions.append(prediction)
    
    # Save predictions
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_predictions, f, indent=2)
    
    return all_predictions


def interactive_prediction(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    visualize_attention: bool = False
):
    """
    Interactive prediction loop.
    
    Args:
        model: Trained QA model.
        tokenizer: Tokenizer for encoding inputs and decoding outputs.
        device: Device to run the model on.
        visualize_attention: Whether to visualize attention weights.
    """
    print("\n=== Interactive Question Answering ===")
    print("Enter a context and a question, and the model will generate an answer.")
    print("Type 'exit' to quit.")
    
    while True:
        # Get context
        context = input("\nContext (or 'exit' to quit): ")
        if context.lower() == "exit":
            break
        
        # Get question
        question = input("Question: ")
        if question.lower() == "exit":
            break
        
        # Generate prediction
        prediction = predict_single_example(
            model=model,
            tokenizer=tokenizer,
            context=context,
            question=question,
            device=device,
            visualize=visualize_attention
        )
        
        # Print prediction
        print(f"\nPrediction: {prediction['prediction']}")
        
        # Visualize attention if requested
        if visualize_attention and "attention_weights" in prediction:
            try:
                import matplotlib.pyplot as plt
                
                # Create attention visualization
                fig = visualize_attention(
                    tokens=prediction["tokens"],
                    attention_weights=prediction["attention_weights"],
                    title="Attention Weights"
                )
                
                # Show visualization
                plt.show()
            except Exception as e:
                print(f"Failed to visualize attention: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with QA models")
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "attn", "transformer"],
                        help="Type of model to use")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                        help="Name of the Hugging Face tokenizer to use")
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["interactive", "file"],
                        help="Prediction mode: interactive or from file")
    parser.add_argument("--input_file", type=str,
                        help="Path to input file (for file mode)")
    parser.add_argument("--output_file", type=str, default="predictions.json",
                        help="Path to save predictions (for file mode)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for prediction")
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="Size of transformer model")
    parser.add_argument("--visualize_attention", action="store_true",
                        help="Whether to visualize attention weights")
    parser.add_argument("--visualization_dir", type=str,
                        help="Directory to save attention visualizations")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = get_tokenizer(args.tokenizer_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    sos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    # Load model
    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        vocab_size=vocab_size,
        device=device,
        model_size=args.model_size,
        padding_idx=pad_token_id,
        sos_token_id=sos_token_id,
        eos_token_id=eos_token_id,
        save_attention=args.visualize_attention
    )
    
    # Run in specified mode
    if args.mode == "interactive":
        interactive_prediction(
            model=model,
            tokenizer=tokenizer,
            device=device,
            visualize_attention=args.visualize_attention
        )
    elif args.mode == "file":
        if not args.input_file:
            parser.error("--input_file is required for file mode")
        
        predict_from_file(
            model=model,
            tokenizer=tokenizer,
            input_file=args.input_file,
            output_file=args.output_file,
            device=device,
            visualize_attention=args.visualize_attention,
            visualization_dir=args.visualization_dir
        )
        
        logger.info(f"Predictions saved to {args.output_file}")
        if args.visualize_attention and args.visualization_dir:
            logger.info(f"Attention visualizations saved to {args.visualization_dir}")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
