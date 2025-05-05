"""
Visualization utilities for QA models.

This module provides functions for visualizing model performance metrics,
training progress, and attention weights.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import pandas as pd
from matplotlib.figure import Figure
from datetime import datetime


def set_plot_style():
    """Set consistent plot style for all visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training and Validation Loss"
) -> Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        save_path: Path to save the plot.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    
    ax.grid(True)
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metric_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Evaluation Metrics"
) -> Figure:
    """
    Plot multiple metric curves.
    
    Args:
        metrics: Dictionary with metric names as keys and lists of values as values.
        save_path: Path to save the plot.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(list(metrics.values())[0]) + 1)
    
    for name, values in metrics.items():
        ax.plot(epochs, values, marker='o', linestyle='-', label=name)
    
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Score')
    ax.legend()
    
    ax.grid(True)
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_progress(
    train_losses: List[float],
    val_losses: List[float],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Progress"
) -> Figure:
    """
    Plot comprehensive training progress with loss and metrics.
    
    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        metrics: Dictionary with metric names as keys and lists of values as values.
        save_path: Path to save the plot.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Plot metrics
    for name, values in metrics.items():
        axes[1].plot(epochs, values, marker='o', linestyle='-', label=name)
    
    axes[1].set_title('Evaluation Metrics')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_attention(
    tokens: List[str],
    attention_weights: np.ndarray,
    title: str = "Attention Weights",
    save_path: Optional[str] = None
) -> Figure:
    """
    Visualize attention weights as a heatmap.
    
    Args:
        tokens: List of input tokens.
        attention_weights: Attention weights [seq_len, seq_len] or [num_heads, seq_len, seq_len].
        title: Plot title.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    set_plot_style()
    
    # Check if the attention weights are from multiple heads
    if len(attention_weights.shape) == 3:
        # Average across heads
        attention_weights = attention_weights.mean(axis=0)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        attention_weights,
        annot=False,
        cmap='viridis',
        ax=ax,
        xticklabels=tokens,
        yticklabels=tokens
    )
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Tokens')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attention_for_sample(
    context: str,
    question: str,
    prediction: str,
    ground_truth: str,
    tokens: List[str],
    attention_weights: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Attention Visualization"
) -> Figure:
    """
    Plot attention weights for a specific QA sample.
    
    Args:
        context: Context text.
        question: Question text.
        prediction: Predicted answer.
        ground_truth: Ground truth answer.
        tokens: List of input tokens.
        attention_weights: Attention weights [seq_len, seq_len] or [num_heads, seq_len, seq_len].
        save_path: Path to save the plot.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    set_plot_style()
    
    # Setup figure and grid
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 6])
    
    # Create text information
    ax1 = fig.add_subplot(gs[0])
    ax1.text(0.01, 0.5, f"Context: {context[:100]}...", fontsize=12, wrap=True)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[1])
    ax2.text(0.01, 0.5, f"Question: {question}", fontsize=12, wrap=True)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[2])
    ax3.text(0.01, 0.5, f"Prediction: {prediction}", fontsize=12, wrap=True, color='blue')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[3])
    color = 'green' if prediction == ground_truth else 'red'
    ax4.text(0.01, 0.5, f"Ground Truth: {ground_truth}", fontsize=12, wrap=True, color=color)
    ax4.axis('off')
    
    # Check if the attention weights are from multiple heads
    if len(attention_weights.shape) == 3:
        # Average across heads
        attention_weights = attention_weights.mean(axis=0)
    
    # Create attention heatmap
    ax5 = fig.add_subplot(gs[4])
    sns.heatmap(
        attention_weights,
        annot=False,
        cmap='viridis',
        ax=ax5,
        xticklabels=tokens,
        yticklabels=tokens
    )
    
    # Set labels
    ax5.set_xlabel('Tokens')
    ax5.set_ylabel('Tokens')
    
    # Rotate x-axis labels
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Set title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_attention_examples(
    examples: List[Dict],
    attentions: List[np.ndarray],
    tokens_list: List[List[str]],
    model_name: str,
    output_dir: str,
    num_examples: int = 5
) -> None:
    """
    Save attention visualizations for multiple examples.
    
    Args:
        examples: List of example dictionaries with 'context', 'question', 'prediction', 'answer'.
        attentions: List of attention weight arrays.
        tokens_list: List of token lists.
        model_name: Name of the model.
        output_dir: Directory to save visualizations.
        num_examples: Maximum number of examples to visualize.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each example
    for i, (example, attention, tokens) in enumerate(zip(examples[:num_examples], attentions[:num_examples], tokens_list[:num_examples])):
        # Create filename
        filename = f"{model_name}_attention_{i+1}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Plot attention
        plot_attention_for_sample(
            context=example['context'],
            question=example['question'],
            prediction=example['prediction'],
            ground_truth=example['answer'],
            tokens=tokens,
            attention_weights=attention,
            save_path=filepath,
            title=f"Attention Visualization - Example {i+1}"
        )


def plot_model_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Model Comparison"
) -> Figure:
    """
    Plot bar chart comparing multiple models on various metrics.
    
    Args:
        model_names: List of model names.
        metrics: Dictionary with metric names as keys and lists of values as values.
        save_path: Path to save the plot.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    set_plot_style()
    
    num_models = len(model_names)
    num_metrics = len(metrics)
    
    # Set width of bars
    bar_width = 0.8 / num_models
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set position of bars on x axis
    positions = np.arange(num_metrics)
    
    # Create bars
    for i, model in enumerate(model_names):
        values = [metrics[metric][i] for metric in metrics.keys()]
        offset = i * bar_width - bar_width * (num_models - 1) / 2
        ax.bar(positions + offset, values, bar_width, label=model)
    
    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(metrics.keys()))
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
