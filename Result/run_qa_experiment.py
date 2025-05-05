"""
Runner script for QA model experiments.

This script provides an easy interface to set up, run, and analyze
experiments with the QA models.
"""

import os
import sys
import argparse
import logging
import json
import subprocess
from datetime import datetime
from typing import Dict, Any

# Import our modules
from results_organizer import create_directory_structure
import results_analysis

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def setup_experiment(base_dir: str, experiment_name: str = None) -> Dict[str, Any]:
    """
    Set up an experiment by creating the directory structure.
    
    Args:
        base_dir: Base directory for the project.
        experiment_name: Name of the experiment.
        
    Returns:
        Dictionary with paths to different directories.
    """
    logger.info("Setting up experiment directory structure...")
    return create_directory_structure(base_dir, experiment_name)


def run_experiment(experiment_dir: str):
    """
    Run a complete experiment using the generated run script.
    
    Args:
        experiment_dir: Path to the experiment directory.
    """
    # Get the path to the run script
    run_script = os.path.join(experiment_dir, "run_experiment.sh")
    
    if not os.path.exists(run_script):
        logger.error(f"Run script not found at {run_script}")
        return
    
    logger.info(f"Running experiment using {run_script}...")
    
    try:
        # Execute the run script
        process = subprocess.Popen(
            run_script,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            logger.error("Experiment failed")
            for line in process.stderr:
                print(line, end='')
        else:
            logger.info("Experiment completed successfully")
    
    except Exception as e:
        logger.error(f"Error running experiment: {e}")


def analyze_results(experiment_dir: str):
    """
    Analyze results from an experiment.
    
    Args:
        experiment_dir: Path to the experiment directory.
    """
    # Get the path to the comparison results
    results_dir = os.path.join(experiment_dir, "evaluation", "comparison")
    output_dir = os.path.join(experiment_dir, "analysis")
    
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found at {results_dir}")
        return
    
    logger.info(f"Analyzing results from {results_dir}...")
    
    # Run analysis
    results_analysis.analyze_results(results_dir, output_dir)
    
    logger.info(f"Analysis completed. Results saved to {output_dir}")


def visualize_attention(experiment_dir: str):
    """
    Create attention visualizations for the models.
    
    Args:
        experiment_dir: Path to the experiment directory.
    """
    # Import attention visualization module
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import attention_visualization
    
    # Get the paths to the model results
    results_dir = os.path.join(experiment_dir, "evaluation", "comparison")
    output_dir = os.path.join(experiment_dir, "plots", "attention")
    
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found at {results_dir}")
        return
    
    logger.info(f"Creating attention visualizations from {results_dir}...")
    
    # Load model results that have attention weights
    model_results = []
    for root, _, files in os.walk(results_dir):
        for filename in files:
            if filename.endswith("_results.json"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r") as f:
                        results = json.load(f)
                        
                        # Check if results have attention weights
                        if "attention_weights" in results or any("attention_weights" in ex for ex in results.get("examples", [])):
                            model_results.append(results)
                            logger.info(f"Loaded attention results from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
    
    if not model_results:
        logger.warning("No model results with attention weights found")
        return
    
    # Process each model's results
    for results in model_results:
        model_name = results.get("model_name", "unknown_model")
        examples = results.get("examples", [])
        
        # Process examples with attention weights
        for i, example in enumerate(examples):
            if i >= 5:  # Limit to first 5 examples
                break
                
            # Check if we have attention weights for this example
            attention_data = {}
            
            # Check for attention weights directly in the example
            if "attention_weights" in example:
                attention_data["attention_weights"] = example["attention_weights"]
            
            # Check for tokens
            if "tokens" in example:
                attention_data["encoder_tokens"] = example["tokens"]
            elif "encoder_tokens" in example:
                attention_data["encoder_tokens"] = example["encoder_tokens"]
            
            # If we have necessary data, create visualization
            if attention_data:
                # Create the dashboard
                dashboard_path = attention_visualization.create_attention_dashboard(
                    model_name=model_name,
                    example_data=example,
                    attention_data=attention_data,
                    output_dir=output_dir,
                    include_animations=False
                )
                
                logger.info(f"Created attention dashboard at {dashboard_path}")
    
    logger.info(f"Attention visualizations created in {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run QA experiments")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory for the project")
    parser.add_argument("--experiment_name", type=str,
                        help="Name of the experiment (optional)")
    parser.add_argument("--setup_only", action="store_true",
                        help="Only set up the experiment without running it")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze results without running the experiment")
    parser.add_argument("--visualize_attention", action="store_true",
                        help="Create attention visualizations")
    parser.add_argument("--existing_experiment", type=str,
                        help="Path to an existing experiment directory (for analyze_only)")
    
    args = parser.parse_args()
    
    # Handle analyze_only mode with existing experiment
    if args.analyze_only and args.existing_experiment:
        if os.path.isdir(args.existing_experiment):
            analyze_results(args.existing_experiment)
            
            if args.visualize_attention:
                visualize_attention(args.existing_experiment)
            
            return
        else:
            logger.error(f"Experiment directory not found: {args.existing_experiment}")
            return
    
    # Setup experiment
    dirs = setup_experiment(args.base_dir, args.experiment_name)
    experiment_dir = dirs["experiment"]
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Exit if setup_only
    if args.setup_only:
        logger.info("Setup complete. Exiting.")
        return
    
    # Run the experiment
    run_experiment(experiment_dir)
    
    # Analyze results
    analyze_results(experiment_dir)
    
    # Create attention visualizations if requested
    if args.visualize_attention:
        visualize_attention(experiment_dir)
    
    logger.info(f"Experiment completed. All results are in {experiment_dir}")


if __name__ == "__main__":
    main() 