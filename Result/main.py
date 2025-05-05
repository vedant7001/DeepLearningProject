"""
Main entry point for the Question Answering system.

This script provides a unified interface for training, evaluation, and inference
across the three model types: LSTM, LSTM with attention, and Transformer.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the QA system."""
    # Create parser
    parser = argparse.ArgumentParser(
        description="Question Answering System with Multiple Models"
    )
    
    # Add subparsers for different actions
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model_type", type=str, required=True,
                              choices=["lstm", "attn", "transformer"],
                              help="Type of model to train")
    train_parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                              help="Name of the Hugging Face tokenizer to use")
    train_parser.add_argument("--output_dir", type=str, default="results",
                              help="Directory to save model and logs")
    train_parser.add_argument("--train_batch_size", type=int, default=16,
                              help="Batch size for training")
    train_parser.add_argument("--eval_batch_size", type=int, default=16,
                              help="Batch size for evaluation")
    train_parser.add_argument("--learning_rate", type=float, default=3e-4,
                              help="Learning rate")
    train_parser.add_argument("--weight_decay", type=float, default=0.01,
                              help="Weight decay for regularization")
    train_parser.add_argument("--num_epochs", type=int, default=10,
                              help="Number of epochs to train")
    train_parser.add_argument("--warmup_steps", type=int, default=1000,
                              help="Number of warmup steps for learning rate scheduler")
    train_parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                              help="Maximum gradient norm for gradient clipping")
    train_parser.add_argument("--use_v2", action="store_true",
                              help="Whether to use SQuAD v2.0 instead of v1.1")
    train_parser.add_argument("--checkpoint_every", type=int, default=1,
                              help="Save checkpoint every N epochs")
    train_parser.add_argument("--log_every", type=int, default=100,
                              help="Log to tensorboard every N steps")
    train_parser.add_argument("--model_size", type=str, default="base",
                              choices=["small", "base", "large"],
                              help="Size of transformer model")
    train_parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5,
                              help="Probability of using teacher forcing (for LSTM models)")
    train_parser.add_argument("--embed_size", type=int, default=256,
                              help="Size of embeddings (for LSTM models)")
    train_parser.add_argument("--hidden_size", type=int, default=512,
                              help="Size of hidden states (for LSTM models)")
    train_parser.add_argument("--num_layers", type=int, default=2,
                              help="Number of layers")
    train_parser.add_argument("--dropout", type=float, default=0.3,
                              help="Dropout probability")
    train_parser.add_argument("--save_attention", action="store_true",
                              help="Whether to save attention weights")
    train_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed")
    
    # Evaluate subparser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--model_configs", type=str, required=True,
                             help="Path to JSON file with model configurations")
    eval_parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                             help="Name of the Hugging Face tokenizer to use")
    eval_parser.add_argument("--data_split", type=str, default="val",
                             choices=["train", "val", "test"],
                             help="Which split to evaluate on")
    eval_parser.add_argument("--batch_size", type=int, default=16,
                             help="Batch size for evaluation")
    eval_parser.add_argument("--output_dir", type=str, default="evaluation_results",
                             help="Directory to save results")
    eval_parser.add_argument("--max_eval_samples", type=int, default=-1,
                             help="Maximum number of samples to evaluate (-1 for all)")
    eval_parser.add_argument("--save_results", action="store_true",
                             help="Whether to save results to disk")
    eval_parser.add_argument("--use_v2", action="store_true",
                             help="Whether to use SQuAD v2.0 instead of v1.1")
    eval_parser.add_argument("--save_attention", action="store_true",
                             help="Whether to save attention visualizations")
    
    # Predict subparser
    predict_parser = subparsers.add_parser("predict", help="Make predictions with a model")
    predict_parser.add_argument("--model_type", type=str, required=True,
                                choices=["lstm", "attn", "transformer"],
                                help="Type of model to use")
    predict_parser.add_argument("--model_path", type=str, required=True,
                                help="Path to model checkpoint")
    predict_parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                                help="Name of the Hugging Face tokenizer to use")
    predict_parser.add_argument("--mode", type=str, default="interactive",
                                choices=["interactive", "file"],
                                help="Prediction mode: interactive or from file")
    predict_parser.add_argument("--input_file", type=str,
                                help="Path to input file (for file mode)")
    predict_parser.add_argument("--output_file", type=str, default="predictions.json",
                                help="Path to save predictions (for file mode)")
    predict_parser.add_argument("--batch_size", type=int, default=16,
                                help="Batch size for prediction")
    predict_parser.add_argument("--model_size", type=str, default="base",
                                choices=["small", "base", "large"],
                                help="Size of transformer model")
    predict_parser.add_argument("--visualize_attention", action="store_true",
                                help="Whether to visualize attention weights")
    predict_parser.add_argument("--visualization_dir", type=str,
                                help="Directory to save attention visualizations")
    
    # Compare subparser
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--results_dir", type=str, required=True,
                                help="Directory containing model results")
    compare_parser.add_argument("--output_dir", type=str, default="comparison_results",
                                help="Directory to save comparison results")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle different actions
    if args.action == "train":
        # Import train module
        from training.train import train
        
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
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seed=args.seed
        )
    
    elif args.action == "evaluate":
        # Import evaluate module
        from training.evaluate import evaluate_all_models
        
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
    
    elif args.action == "predict":
        # Import predict module
        from training.predict import load_model, interactive_prediction, predict_from_file
        import torch
        from utils.tokenization import get_tokenizer
        
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
    
    elif args.action == "compare":
        # Import compare module
        from training.evaluate import compare_models
        import glob
        import json
        
        # Find result files
        result_files = glob.glob(os.path.join(args.results_dir, "**/results.json"), recursive=True)
        
        if not result_files:
            logger.error(f"No result files found in {args.results_dir}")
            return
        
        # Load results
        all_results = []
        for file_path in result_files:
            with open(file_path, "r") as f:
                results = json.load(f)
                all_results.append(results)
        
        # Compare models
        compare_models(all_results, args.output_dir)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
