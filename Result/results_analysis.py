"""
Results analysis and visualization module.

This module provides functions for analyzing and visualizing results from
different QA models, generating comprehensive plots and reports.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from tabulate import tabulate
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load model results from files in the results directory.
    
    Args:
        results_dir: Directory containing result files.
        
    Returns:
        List of model result dictionaries.
    """
    # Find all result files (recursive search)
    result_files = []
    for root, _, files in os.walk(results_dir):
        for filename in files:
            if filename.endswith("_results.json"):
                result_files.append(os.path.join(root, filename))
    
    logger.info(f"Found {len(result_files)} result files")
    
    # Load results from files
    all_results = []
    for file_path in result_files:
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
                model_name = results.get("model_name", os.path.basename(file_path).replace("_results.json", ""))
                results["model_name"] = model_name
                all_results.append(results)
                logger.info(f"Loaded results for model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return all_results


def create_metrics_table(model_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a table of metrics for all models.
    
    Args:
        model_results: List of model result dictionaries.
        
    Returns:
        DataFrame with metrics for all models.
    """
    # Extract model names and metrics
    data = []
    metrics = set()
    
    for results in model_results:
        model_name = results["model_name"]
        model_metrics = results.get("metrics", {})
        
        # Add all metrics to the set
        metrics.update(model_metrics.keys())
        
        # Create row with model name and metrics
        row = {"Model": model_name}
        row.update(model_metrics)
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort columns to ensure consistent order
    metric_cols = sorted(list(metrics))
    columns = ["Model"] + metric_cols
    
    # Reorder columns and return
    return df[columns].sort_values("Model")


def plot_metrics_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot comparison of metrics across models.
    
    Args:
        df: DataFrame with metrics for all models.
        output_dir: Directory to save plots.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metrics (all columns except 'Model')
    metrics = df.columns.tolist()
    metrics.remove("Model")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Create comparison plot for all metrics
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Reshape DataFrame for plotting
    plot_df = pd.melt(df, id_vars=["Model"], value_vars=metrics, 
                      var_name="Metric", value_name="Score")
    
    # Plot grouped bar chart
    sns.barplot(x="Metric", y="Score", hue="Model", data=plot_df, ax=ax)
    
    # Customize plot
    ax.set_title("Comparison of Metrics Across Models", fontsize=16)
    ax.set_xlabel("Metric", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    fig.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Create individual plots for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by metric value
        sorted_df = df.sort_values(metric, ascending=False)
        
        # Plot bar chart
        sns.barplot(x="Model", y=metric, data=sorted_df, ax=ax, palette="viridis")
        
        # Add value labels
        for i, v in enumerate(sorted_df[metric]):
            ax.text(i, v + 0.5, f"{v:.2f}", ha="center", fontsize=12)
        
        # Customize plot
        ax.set_title(f"Comparison of {metric} Across Models", fontsize=16)
        ax.set_xlabel("Model", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        plt.xticks(rotation=45, ha="right")
        
        # Adjust y-axis to include space for labels
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.1)
        
        plt.tight_layout()
        
        # Save plot
        metric_filename = metric.replace(" ", "_").lower()
        fig.savefig(os.path.join(output_dir, f"{metric_filename}_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def analyze_example_predictions(model_results: List[Dict[str, Any]], output_dir: str):
    """
    Analyze and visualize example predictions from each model.
    
    Args:
        model_results: List of model result dictionaries.
        output_dir: Directory to save analysis results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract examples from each model
    all_examples = []
    
    for results in model_results:
        model_name = results["model_name"]
        examples = results.get("examples", [])
        
        for example in examples:
            # Add model name to example
            example["model"] = model_name
            all_examples.append(example)
    
    # Group examples by question/context
    example_groups = {}
    
    for example in all_examples:
        # Create a key from the question and context (first 100 chars)
        key = (example.get("question", ""), example.get("context", "")[:100])
        
        if key not in example_groups:
            example_groups[key] = []
        
        example_groups[key].append(example)
    
    # Select groups that have predictions from multiple models
    comparative_examples = []
    
    for key, group in example_groups.items():
        if len(group) > 1 and len(set(ex["model"] for ex in group)) > 1:
            comparative_examples.append(group)
    
    # Sort by number of models and then by average F1 score
    comparative_examples.sort(
        key=lambda group: (
            len(set(ex["model"] for ex in group)), 
            sum(ex.get("f1", 0) for ex in group) / len(group)
        ),
        reverse=True
    )
    
    # Take the top examples (maximum 20)
    top_examples = comparative_examples[:min(20, len(comparative_examples))]
    
    # Create a report with comparative examples
    report_path = os.path.join(output_dir, "comparative_examples.html")
    
    with open(report_path, "w") as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<title>Comparative Example Predictions</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("h1 { color: #333; }\n")
        f.write("h2 { color: #555; margin-top: 30px; }\n")
        f.write(".example { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }\n")
        f.write(".context { background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px; }\n")
        f.write(".question { font-weight: bold; margin-bottom: 10px; }\n")
        f.write(".answer { color: #006600; margin-bottom: 10px; }\n")
        f.write(".prediction { margin-bottom: 5px; }\n")
        f.write(".metrics { color: #666; font-style: italic; }\n")
        f.write(".good { color: #006600; }\n")
        f.write(".bad { color: #660000; }\n")
        f.write("table { border-collapse: collapse; width: 100%; }\n")
        f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        f.write("th { background-color: #f2f2f2; }\n")
        f.write("</style>\n</head>\n<body>\n")
        
        f.write("<h1>Comparative Example Predictions</h1>\n")
        
        for i, group in enumerate(top_examples):
            f.write(f"<div class='example'>\n")
            f.write(f"<h2>Example {i+1}</h2>\n")
            
            # Get context and question (same for all models in the group)
            context = group[0].get("context", "")
            question = group[0].get("question", "")
            answer = group[0].get("answer", "")
            
            f.write(f"<div class='context'>Context: {context}</div>\n")
            f.write(f"<div class='question'>Question: {question}</div>\n")
            f.write(f"<div class='answer'>Answer: {answer}</div>\n")
            
            # Create table for model predictions
            f.write("<table>\n")
            f.write("<tr><th>Model</th><th>Prediction</th><th>F1 Score</th><th>Exact Match</th></tr>\n")
            
            # Sort group by F1 score
            sorted_group = sorted(group, key=lambda ex: ex.get("f1", 0), reverse=True)
            
            for example in sorted_group:
                model = example.get("model", "Unknown")
                prediction = example.get("prediction", "")
                f1 = example.get("f1", 0)
                exact_match = example.get("exact_match", 0)
                
                # Determine class for metrics
                f1_class = "good" if f1 >= 70 else "bad"
                em_class = "good" if exact_match >= 70 else "bad"
                
                f.write(f"<tr>\n")
                f.write(f"<td>{model}</td>\n")
                f.write(f"<td>{prediction}</td>\n")
                f.write(f"<td class='{f1_class}'>{f1:.2f}</td>\n")
                f.write(f"<td class='{em_class}'>{exact_match:.2f}</td>\n")
                f.write(f"</tr>\n")
            
            f.write("</table>\n")
            f.write("</div>\n")
        
        f.write("</body>\n</html>")
    
    logger.info(f"Comparative examples report saved to {report_path}")


def create_failure_analysis(model_results: List[Dict[str, Any]], output_dir: str):
    """
    Create a failure analysis report for each model.
    
    Args:
        model_results: List of model result dictionaries.
        output_dir: Directory to save analysis results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a report for each model
    for results in model_results:
        model_name = results["model_name"]
        predictions = results.get("predictions", [])
        targets = results.get("targets", [])
        questions = results.get("questions", [])
        contexts = results.get("contexts", [])
        
        # If any of these are missing, skip this model
        if not (predictions and targets and questions and contexts):
            logger.warning(f"Skipping failure analysis for {model_name}: missing data")
            continue
        
        # Calculate F1 scores for each example
        f1_scores = []
        exact_matches = []
        
        from utils.metrics import compute_f1, compute_exact_match
        
        for pred, target in zip(predictions, targets):
            f1 = compute_f1(pred, target) * 100
            em = compute_exact_match(pred, target) * 100
            
            f1_scores.append(f1)
            exact_matches.append(em)
        
        # Create a DataFrame with all examples
        data = {
            "question": questions,
            "context": [c[:200] + "..." if len(c) > 200 else c for c in contexts],
            "target": targets,
            "prediction": predictions,
            "f1": f1_scores,
            "exact_match": exact_matches
        }
        
        df = pd.DataFrame(data)
        
        # Sort by F1 score (ascending to see failures first)
        df = df.sort_values("f1")
        
        # Create an HTML report
        report_path = os.path.join(output_dir, f"{model_name}_failure_analysis.html")
        
        with open(report_path, "w") as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write(f"<title>{model_name} - Failure Analysis</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("h1 { color: #333; }\n")
            f.write("h2 { color: #555; margin-top: 30px; }\n")
            f.write(".example { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }\n")
            f.write(".worst { background-color: #ffeeee; }\n")
            f.write(".bad { background-color: #fff6ee; }\n")
            f.write(".good { background-color: #eeffee; }\n")
            f.write(".perfect { background-color: #e6ffe6; }\n")
            f.write(".context { background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px; }\n")
            f.write(".question { font-weight: bold; margin-bottom: 10px; }\n")
            f.write(".answer { color: #006600; margin-bottom: 10px; }\n")
            f.write(".prediction { margin-bottom: 5px; }\n")
            f.write(".metrics { color: #666; font-style: italic; }\n")
            f.write("</style>\n</head>\n<body>\n")
            
            f.write(f"<h1>{model_name} - Failure Analysis</h1>\n")
            
            # Write summary statistics
            f.write("<h2>Summary</h2>\n")
            f.write("<ul>\n")
            f.write(f"<li>Total examples: {len(df)}</li>\n")
            f.write(f"<li>Average F1 score: {df['f1'].mean():.2f}</li>\n")
            f.write(f"<li>Average Exact Match: {df['exact_match'].mean():.2f}</li>\n")
            f.write(f"<li>Perfect predictions (F1=100): {sum(df['f1'] == 100)}</li>\n")
            f.write(f"<li>Good predictions (F1≥70): {sum(df['f1'] >= 70)}</li>\n")
            f.write(f"<li>Poor predictions (F1<50): {sum(df['f1'] < 50)}</li>\n")
            f.write(f"<li>Complete failures (F1=0): {sum(df['f1'] == 0)}</li>\n")
            f.write("</ul>\n")
            
            # Show worst examples (bottom 10)
            f.write("<h2>Worst Examples</h2>\n")
            
            worst = df.head(10)
            for i, row in worst.iterrows():
                example_class = "worst" if row["f1"] == 0 else "bad"
                f.write(f"<div class='example {example_class}'>\n")
                f.write(f"<div class='context'>Context: {row['context']}</div>\n")
                f.write(f"<div class='question'>Question: {row['question']}</div>\n")
                f.write(f"<div class='answer'>Target: {row['target']}</div>\n")
                f.write(f"<div class='prediction'>Prediction: {row['prediction']}</div>\n")
                f.write(f"<div class='metrics'>F1: {row['f1']:.2f} | Exact Match: {row['exact_match']:.2f}</div>\n")
                f.write("</div>\n")
            
            # Show best examples (top 10)
            f.write("<h2>Best Examples</h2>\n")
            
            best = df.tail(10).iloc[::-1]
            for i, row in best.iterrows():
                example_class = "perfect" if row["f1"] == 100 else "good"
                f.write(f"<div class='example {example_class}'>\n")
                f.write(f"<div class='context'>Context: {row['context']}</div>\n")
                f.write(f"<div class='question'>Question: {row['question']}</div>\n")
                f.write(f"<div class='answer'>Target: {row['target']}</div>\n")
                f.write(f"<div class='prediction'>Prediction: {row['prediction']}</div>\n")
                f.write(f"<div class='metrics'>F1: {row['f1']:.2f} | Exact Match: {row['exact_match']:.2f}</div>\n")
                f.write("</div>\n")
            
            f.write("</body>\n</html>")
        
        logger.info(f"Failure analysis report for {model_name} saved to {report_path}")


def generate_summary_dashboard(model_results: List[Dict[str, Any]], output_dir: str):
    """
    Generate a summary dashboard of results from all models.
    
    Args:
        model_results: List of model result dictionaries.
        output_dir: Directory to save the dashboard.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metrics DataFrame
    metrics_df = create_metrics_table(model_results)
    
    # Create dashboard HTML
    dashboard_path = os.path.join(output_dir, "results_dashboard.html")
    
    with open(dashboard_path, "w") as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<title>QA Models Results Dashboard</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("h1 { color: #333; }\n")
        f.write("h2 { color: #555; margin-top: 30px; }\n")
        f.write("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n")
        f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        f.write("th { background-color: #f2f2f2; }\n")
        f.write(".chart-container { width: 100%; margin-bottom: 30px; }\n")
        f.write(".chart-image { max-width: 100%; border: 1px solid #ddd; }\n")
        f.write(".links-container { margin-top: 30px; }\n")
        f.write(".link-item { margin-bottom: 10px; }\n")
        f.write("</style>\n</head>\n<body>\n")
        
        f.write("<h1>Question Answering Models - Results Dashboard</h1>\n")
        
        # Summary table
        f.write("<h2>Model Performance Metrics</h2>\n")
        f.write("<table>\n")
        
        # Write table header
        f.write("<tr>\n")
        for column in metrics_df.columns:
            f.write(f"<th>{column}</th>\n")
        f.write("</tr>\n")
        
        # Write table rows
        for _, row in metrics_df.iterrows():
            f.write("<tr>\n")
            for column in metrics_df.columns:
                value = row[column]
                if isinstance(value, (int, float)):
                    f.write(f"<td>{value:.2f}</td>\n")
                else:
                    f.write(f"<td>{value}</td>\n")
            f.write("</tr>\n")
        
        f.write("</table>\n")
        
        # Charts
        f.write("<h2>Performance Comparison Charts</h2>\n")
        
        # Metrics comparison chart
        f.write("<div class='chart-container'>\n")
        f.write("<h3>Overall Metrics Comparison</h3>\n")
        f.write(f"<img class='chart-image' src='metrics_comparison.png' alt='Metrics Comparison'>\n")
        f.write("</div>\n")
        
        # Individual metric charts
        metrics = [col for col in metrics_df.columns if col != "Model"]
        for metric in metrics:
            metric_filename = metric.replace(" ", "_").lower()
            f.write("<div class='chart-container'>\n")
            f.write(f"<h3>{metric} Comparison</h3>\n")
            f.write(f"<img class='chart-image' src='{metric_filename}_comparison.png' alt='{metric} Comparison'>\n")
            f.write("</div>\n")
        
        # Links to detailed reports
        f.write("<div class='links-container'>\n")
        f.write("<h2>Detailed Analysis Reports</h2>\n")
        
        # Comparative examples
        f.write("<div class='link-item'>\n")
        f.write("<a href='comparative_examples.html'>Comparative Examples Analysis</a> - Side-by-side comparison of model predictions on the same examples\n")
        f.write("</div>\n")
        
        # Failure analysis for each model
        f.write("<h3>Failure Analysis Reports</h3>\n")
        for results in model_results:
            model_name = results["model_name"]
            f.write("<div class='link-item'>\n")
            f.write(f"<a href='{model_name}_failure_analysis.html'>{model_name} Failure Analysis</a> - Detailed analysis of prediction errors\n")
            f.write("</div>\n")
        
        f.write("</div>\n")
        
        f.write("</body>\n</html>")
    
    logger.info(f"Results dashboard saved to {dashboard_path}")


def analyze_results(results_dir: str, output_dir: str):
    """
    Analyze results and generate visualizations.
    
    Args:
        results_dir: Directory containing model results.
        output_dir: Directory to save analysis results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model results
    model_results = load_results(results_dir)
    
    if not model_results:
        logger.error(f"No results found in {results_dir}")
        return
    
    # Create metrics table
    metrics_df = create_metrics_table(model_results)
    
    # Save metrics table as CSV
    metrics_path = os.path.join(output_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics summary saved to {metrics_path}")
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_df, output_dir)
    logger.info("Metrics comparison plots created")
    
    # Analyze example predictions
    analyze_example_predictions(model_results, output_dir)
    logger.info("Example predictions analysis completed")
    
    # Create failure analysis
    create_failure_analysis(model_results, output_dir)
    logger.info("Failure analysis completed")
    
    # Generate summary dashboard
    generate_summary_dashboard(model_results, output_dir)
    logger.info("Summary dashboard created")
    
    # Print summary
    logger.info("\nResults Analysis Complete!")
    logger.info(f"Summary dashboard: {os.path.join(output_dir, 'results_dashboard.html')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze QA model results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing model results")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    analyze_results(args.results_dir, args.output_dir) 