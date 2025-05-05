"""
Attention visualization utilities for QA models.

This module provides specialized functions for visualizing attention
weights from different QA models, including multi-head attention
from transformers and Bahdanau attention from LSTM models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from matplotlib.figure import Figure
import math
from matplotlib.colors import LinearSegmentedColormap


def set_visualization_style():
    """Set consistent style for attention visualizations."""
    sns.set_style("white")
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


def create_attention_heatmap(
    attention_weights: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str = "Attention Weights",
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    show_colorbar: bool = True,
    show_grid: bool = True
) -> Figure:
    """
    Create a heatmap visualization of attention weights.
    
    Args:
        attention_weights: Attention weights matrix [m, n].
        x_labels: Labels for the x-axis (columns).
        y_labels: Labels for the y-axis (rows).
        title: Title for the plot.
        figsize: Figure size (width, height).
        cmap: Colormap to use.
        save_path: Path to save the visualization.
        show_colorbar: Whether to show a colorbar.
        show_grid: Whether to show grid lines.
        
    Returns:
        Matplotlib figure.
    """
    # Set style
    set_visualization_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    heatmap = sns.heatmap(
        attention_weights,
        annot=False,
        cmap=cmap,
        cbar=show_colorbar,
        ax=ax,
        xticklabels=x_labels,
        yticklabels=y_labels,
        square=True
    )
    
    # Add grid
    if show_grid:
        ax.set_xticks(np.arange(attention_weights.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(attention_weights.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Set title and labels
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_multihead_attention(
    attention_weights: Union[np.ndarray, torch.Tensor],
    tokens: List[str],
    head_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
    average_heads: bool = False,
    title: str = "Multi-Head Attention",
    save_path: Optional[str] = None
) -> Figure:
    """
    Visualize multi-head attention weights from a transformer model.
    
    Args:
        attention_weights: Attention weights with shape [num_layers, num_heads, seq_len, seq_len]
                           or [num_heads, seq_len, seq_len] if layer_idx is not None.
        tokens: List of tokens corresponding to the sequence.
        head_idx: Index of the head to visualize (None for all heads).
        layer_idx: Index of the layer to visualize (None for last layer).
        average_heads: Whether to average across heads.
        title: Title for the plot.
        save_path: Path to save the visualization.
        
    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Extract the layer to visualize
    if len(attention_weights.shape) == 4:  # [num_layers, num_heads, seq_len, seq_len]
        if layer_idx is None:
            layer_idx = attention_weights.shape[0] - 1  # Last layer
        layer_attention = attention_weights[layer_idx]
    else:  # [num_heads, seq_len, seq_len]
        layer_attention = attention_weights
    
    # Handle the head dimension
    if head_idx is not None:
        # Visualize a single head
        attn_to_plot = layer_attention[head_idx]
        title = f"{title} (Layer {layer_idx}, Head {head_idx})"
    elif average_heads:
        # Average across heads
        attn_to_plot = layer_attention.mean(axis=0)
        title = f"{title} (Layer {layer_idx}, Averaged across heads)"
    else:
        # Visualize all heads in a grid
        num_heads = layer_attention.shape[0]
        nrows = int(math.sqrt(num_heads))
        ncols = math.ceil(num_heads / nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 14))
        fig.suptitle(f"{title} (Layer {layer_idx})", fontsize=18)
        
        for i, ax in enumerate(axes.flat):
            if i < num_heads:
                sns.heatmap(
                    layer_attention[i],
                    annot=False,
                    cmap="viridis",
                    cbar=False,
                    ax=ax,
                    xticklabels=tokens if i >= num_heads - ncols else [],
                    yticklabels=tokens if i % ncols == 0 else [],
                    square=True
                )
                ax.set_title(f"Head {i}")
                
                # Rotate labels if they are displayed
                if i >= num_heads - ncols:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save if path is provided
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # Create the heatmap for a single head or averaged heads
    return create_attention_heatmap(
        attention_weights=attn_to_plot,
        x_labels=tokens,
        y_labels=tokens,
        title=title,
        save_path=save_path
    )


def visualize_lstm_attention(
    attention_weights: Union[np.ndarray, torch.Tensor],
    source_tokens: List[str],
    target_tokens: List[str],
    title: str = "LSTM Attention",
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = "YlOrRd",
    save_path: Optional[str] = None
) -> Figure:
    """
    Visualize attention weights from an LSTM encoder-decoder with attention.
    
    Args:
        attention_weights: Attention weights with shape [target_len, source_len].
        source_tokens: Tokens from the source sequence (encoder).
        target_tokens: Tokens from the target sequence (decoder).
        title: Title for the plot.
        figsize: Figure size (width, height).
        cmap: Colormap to use.
        save_path: Path to save the visualization.
        
    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Ensure we have the right shape
    if len(attention_weights.shape) != 2:
        raise ValueError(f"Expected attention_weights to have shape [target_len, source_len], "
                         f"got {attention_weights.shape}")
    
    # Set style
    set_visualization_style()
    
    # Create a custom colormap with more contrast
    customcmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#FFFFFF", "#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84", 
                        "#FC8D59", "#EF6548", "#D7301F", "#B30000", "#7F0000"]
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    heatmap = sns.heatmap(
        attention_weights,
        annot=True,
        fmt=".2f",
        cmap=cmap if cmap != "custom" else customcmap,
        cbar=True,
        ax=ax,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        square=True
    )
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Source Tokens (Input)")
    ax.set_ylabel("Target Tokens (Output)")
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_encoder_decoder_attention(
    attention_weights: Union[np.ndarray, torch.Tensor],
    context_tokens: List[str],
    question_tokens: List[str],
    answer_tokens: List[str],
    title: str = "Encoder-Decoder Attention",
    save_path: Optional[str] = None
) -> Figure:
    """
    Visualize attention between encoder (context+question) and decoder (answer).
    
    Args:
        attention_weights: Attention weights [answer_len, context_len+question_len].
        context_tokens: Tokens from the context.
        question_tokens: Tokens from the question.
        answer_tokens: Tokens from the answer.
        title: Title for the plot.
        save_path: Path to save the visualization.
        
    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Combine context and question tokens
    encoder_tokens = context_tokens + question_tokens
    
    # Ensure we have the right shape
    if attention_weights.shape[1] != len(encoder_tokens):
        # Try to handle the case where special tokens might be included/excluded
        if abs(attention_weights.shape[1] - len(encoder_tokens)) <= 3:
            # Adjust token list or attention weights to match
            if attention_weights.shape[1] > len(encoder_tokens):
                # Add placeholder tokens
                missing = attention_weights.shape[1] - len(encoder_tokens)
                encoder_tokens = ["[PAD]"] * missing + encoder_tokens
            else:
                # Truncate attention weights
                attention_weights = attention_weights[:, -len(encoder_tokens):]
        else:
            raise ValueError(f"Attention weights shape {attention_weights.shape[1]} doesn't match "
                             f"encoder tokens length {len(encoder_tokens)}")
    
    # Set style
    set_visualization_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot heatmap
    heatmap = sns.heatmap(
        attention_weights,
        annot=False,
        cmap="YlOrRd",
        cbar=True,
        ax=ax,
        xticklabels=encoder_tokens,
        yticklabels=answer_tokens,
        square=False
    )
    
    # Add a vertical line to separate context and question
    if len(context_tokens) < len(encoder_tokens):
        ax.axvline(x=len(context_tokens), color='blue', linestyle='-', linewidth=2)
        
        # Add labels for context and question regions
        ctx_center = len(context_tokens) / 2
        q_center = len(context_tokens) + len(question_tokens) / 2
        
        ax.text(ctx_center, -0.5, "Context", ha="center", va="top", color="blue", fontsize=12)
        ax.text(q_center, -0.5, "Question", ha="center", va="top", color="green", fontsize=12)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Encoder Tokens (Context + Question)")
    ax.set_ylabel("Decoder Tokens (Answer)")
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_attention_animation(
    attention_sequence: Union[np.ndarray, torch.Tensor],
    source_tokens: List[str],
    target_tokens: List[str],
    title: str = "Attention Over Time",
    save_path: Optional[str] = None
) -> None:
    """
    Create an animation of attention weights changing over decoding time steps.
    
    Args:
        attention_sequence: Sequence of attention weights [time_steps, source_len].
        source_tokens: Tokens from the source sequence.
        target_tokens: Tokens from the target sequence.
        title: Title for the animation.
        save_path: Path to save the animation.
        
    Note:
        This function requires matplotlib's animation functionality and ffmpeg.
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        print("Failed to import matplotlib.animation. Make sure matplotlib is installed.")
        return
    
    # Convert to numpy if tensor
    if isinstance(attention_sequence, torch.Tensor):
        attention_sequence = attention_sequence.detach().cpu().numpy()
    
    # Set style
    set_visualization_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        time_step = min(frame, len(attention_sequence) - 1)
        attention = attention_sequence[time_step]
        
        # Get the current token being generated
        current_token = target_tokens[time_step] if time_step < len(target_tokens) else "<end>"
        
        # Plot heatmap for current time step
        sns.heatmap(
            attention.reshape(1, -1),
            annot=False,
            cmap="YlOrRd",
            cbar=True,
            ax=ax,
            xticklabels=source_tokens,
            yticklabels=[current_token],
            square=False
        )
        
        # Set title and labels
        ax.set_title(f"{title} - Generating: {current_token}")
        ax.set_xlabel("Source Tokens")
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(attention_sequence) + 5,  # Extra frames at the end
        interval=500,  # ms between frames
        repeat_delay=1000  # ms before repeating
    )
    
    # Save animation if path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        try:
            anim.save(save_path, writer='ffmpeg')
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            print("Make sure ffmpeg is installed and in your PATH.")
    
    plt.close(fig)


def create_attention_dashboard(
    model_name: str,
    example_data: Dict[str, Any],
    attention_data: Dict[str, Any],
    output_dir: str,
    include_animations: bool = False
) -> str:
    """
    Create a comprehensive attention visualization dashboard for a single example.
    
    Args:
        model_name: Name of the model.
        example_data: Dictionary with example data (context, question, prediction, etc.).
        attention_data: Dictionary with attention weights and related data.
        output_dir: Directory to save the dashboard and associated files.
        include_animations: Whether to include attention animations.
        
    Returns:
        Path to the generated HTML dashboard.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract example data
    context = example_data.get("context", "")
    question = example_data.get("question", "")
    prediction = example_data.get("prediction", "")
    target = example_data.get("answer", "")
    example_id = example_data.get("id", "unknown")
    
    # Create paths for figures
    example_dir = os.path.join(output_dir, f"example_{example_id}")
    os.makedirs(example_dir, exist_ok=True)
    
    # Extract tokens and attention weights
    encoder_tokens = attention_data.get("encoder_tokens", [])
    decoder_tokens = attention_data.get("decoder_tokens", [])
    
    figures = {}
    
    # Generate visualizations based on model type
    if "transformer" in model_name.lower():
        # For transformer models
        if "self_attention" in attention_data:
            # Encoder self-attention
            enc_self_attn = attention_data["self_attention"]
            enc_fig_path = os.path.join(example_dir, "encoder_self_attention.png")
            
            # Visualize all heads in the last layer
            fig = visualize_multihead_attention(
                attention_weights=enc_self_attn,
                tokens=encoder_tokens,
                title=f"{model_name}: Encoder Self-Attention",
                save_path=enc_fig_path
            )
            figures["encoder_self_attention"] = enc_fig_path
            plt.close(fig)
            
            # Visualize averaged heads
            avg_enc_fig_path = os.path.join(example_dir, "encoder_self_attention_avg.png")
            fig = visualize_multihead_attention(
                attention_weights=enc_self_attn,
                tokens=encoder_tokens,
                average_heads=True,
                title=f"{model_name}: Encoder Self-Attention (Averaged)",
                save_path=avg_enc_fig_path
            )
            figures["encoder_self_attention_avg"] = avg_enc_fig_path
            plt.close(fig)
        
        if "cross_attention" in attention_data:
            # Cross-attention
            cross_attn = attention_data["cross_attention"]
            cross_fig_path = os.path.join(example_dir, "cross_attention.png")
            
            # Get context and question tokens
            context_tokens = encoder_tokens[:len(encoder_tokens)//2]  # Approximate if not available
            question_tokens = encoder_tokens[len(encoder_tokens)//2:]  # Approximate if not available
            
            if "context_tokens" in attention_data:
                context_tokens = attention_data["context_tokens"]
            
            if "question_tokens" in attention_data:
                question_tokens = attention_data["question_tokens"]
            
            # Visualize encoder-decoder attention
            fig = visualize_encoder_decoder_attention(
                attention_weights=cross_attn,
                context_tokens=context_tokens,
                question_tokens=question_tokens,
                answer_tokens=decoder_tokens,
                title=f"{model_name}: Encoder-Decoder Attention",
                save_path=cross_fig_path
            )
            figures["cross_attention"] = cross_fig_path
            plt.close(fig)
    elif "attn" in model_name.lower():
        # For LSTM with attention models
        if "attention_weights" in attention_data:
            attn = attention_data["attention_weights"]
            attn_fig_path = os.path.join(example_dir, "lstm_attention.png")
            
            # Visualize LSTM attention
            if len(decoder_tokens) > 0:
                fig = visualize_lstm_attention(
                    attention_weights=attn,
                    source_tokens=encoder_tokens,
                    target_tokens=decoder_tokens,
                    title=f"{model_name}: LSTM Attention",
                    save_path=attn_fig_path
                )
                figures["lstm_attention"] = attn_fig_path
                plt.close(fig)
            
            # Create attention over time animation if requested
            if include_animations and "attention_sequence" in attention_data:
                anim_path = os.path.join(example_dir, "attention_animation.mp4")
                
                create_attention_animation(
                    attention_sequence=attention_data["attention_sequence"],
                    source_tokens=encoder_tokens,
                    target_tokens=decoder_tokens,
                    title=f"{model_name}: Attention Over Time",
                    save_path=anim_path
                )
                
                figures["attention_animation"] = anim_path
    
    # Create HTML dashboard
    dashboard_path = os.path.join(example_dir, "attention_dashboard.html")
    
    with open(dashboard_path, "w") as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write(f"<title>{model_name} - Attention Visualization</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n")
        f.write("h1 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }\n")
        f.write("h2 { color: #555; margin-top: 20px; }\n")
        f.write("h3 { color: #666; }\n")
        f.write(".container { max-width: 1200px; margin: 0 auto; }\n")
        f.write(".example-info { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }\n")
        f.write(".context { background-color: #f0f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px; }\n")
        f.write(".question { font-weight: bold; }\n")
        f.write(".prediction { color: #0066cc; }\n")
        f.write(".target { color: #006600; }\n")
        f.write(".attention-container { margin-top: 30px; }\n")
        f.write(".attention-viz { margin-bottom: 30px; }\n")
        f.write(".attention-img { max-width: 100%; border: 1px solid #ddd; box-shadow: 0 0 10px rgba(0,0,0,0.1); }\n")
        f.write(".match { background-color: #e6ffe6; }\n")
        f.write(".mismatch { background-color: #fff0f0; }\n")
        f.write(".token-vis { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 20px; }\n")
        f.write(".token-table { width: 100%; border-collapse: collapse; margin-top: 10px; }\n")
        f.write(".token-table th, .token-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        f.write(".token-table th { background-color: #f2f2f2; }\n")
        f.write("</style>\n</head>\n<body>\n")
        
        f.write("<div class='container'>\n")
        
        # Header
        f.write(f"<h1>{model_name} - Attention Visualization</h1>\n")
        
        # Example information
        f.write("<div class='example-info'>\n")
        f.write(f"<h2>Example {example_id}</h2>\n")
        f.write(f"<div class='context'><strong>Context:</strong> {context}</div>\n")
        f.write(f"<div class='question'><strong>Question:</strong> {question}</div>\n")
        
        # Determine if prediction matches target
        match_class = "match" if prediction.lower() == target.lower() else "mismatch"
        
        f.write(f"<div class='prediction'><strong>Prediction:</strong> <span class='{match_class}'>{prediction}</span></div>\n")
        f.write(f"<div class='target'><strong>Target:</strong> {target}</div>\n")
        f.write("</div>\n")
        
        # Token information
        f.write("<div class='token-vis'>\n")
        f.write("<h2>Tokenization</h2>\n")
        
        # Encoder tokens
        f.write("<h3>Encoder Tokens</h3>\n")
        f.write("<table class='token-table'>\n")
        f.write("<tr><th>Position</th><th>Token</th></tr>\n")
        
        for i, token in enumerate(encoder_tokens):
            f.write(f"<tr><td>{i}</td><td>{token}</td></tr>\n")
        
        f.write("</table>\n")
        
        # Decoder tokens if available
        if decoder_tokens:
            f.write("<h3>Decoder Tokens</h3>\n")
            f.write("<table class='token-table'>\n")
            f.write("<tr><th>Position</th><th>Token</th></tr>\n")
            
            for i, token in enumerate(decoder_tokens):
                f.write(f"<tr><td>{i}</td><td>{token}</td></tr>\n")
            
            f.write("</table>\n")
        
        f.write("</div>\n")
        
        # Attention visualizations
        f.write("<div class='attention-container'>\n")
        f.write("<h2>Attention Visualizations</h2>\n")
        
        for viz_name, viz_path in figures.items():
            # Get the relative path for the HTML file
            rel_path = os.path.relpath(viz_path, example_dir)
            
            f.write("<div class='attention-viz'>\n")
            viz_title = viz_name.replace("_", " ").title()
            f.write(f"<h3>{viz_title}</h3>\n")
            
            if viz_path.endswith(".mp4"):
                # Video for animation
                f.write(f"<video width='100%' controls>\n")
                f.write(f"  <source src='{rel_path}' type='video/mp4'>\n")
                f.write(f"  Your browser does not support the video tag.\n")
                f.write(f"</video>\n")
            else:
                # Image for static visualization
                f.write(f"<img class='attention-img' src='{rel_path}' alt='{viz_title}'>\n")
            
            f.write("</div>\n")
        
        f.write("</div>\n")
        
        f.write("</div>\n")
        f.write("</body>\n</html>")
    
    return dashboard_path


def main():
    """Example usage of the attention visualization functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize attention weights")
    parser.add_argument("--attention_file", type=str, required=True,
                        help="Path to a numpy file containing attention weights")
    parser.add_argument("--model_type", type=str, choices=["transformer", "lstm", "attn"],
                        required=True, help="Type of model")
    parser.add_argument("--output_dir", type=str, default="attention_viz",
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Load attention weights
    attention_weights = np.load(args.attention_file)
    
    # Create sample tokens
    encoder_tokens = [f"src_{i}" for i in range(10)]
    decoder_tokens = [f"tgt_{i}" for i in range(5)]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize based on model type
    if args.model_type == "transformer":
        # Assume multi-head attention from transformer
        fig = visualize_multihead_attention(
            attention_weights=attention_weights,
            tokens=encoder_tokens,
            title="Transformer Self-Attention",
            save_path=os.path.join(args.output_dir, "transformer_attention.png")
        )
    elif args.model_type == "attn":
        # Assume LSTM with attention
        fig = visualize_lstm_attention(
            attention_weights=attention_weights,
            source_tokens=encoder_tokens,
            target_tokens=decoder_tokens,
            title="LSTM Attention",
            save_path=os.path.join(args.output_dir, "lstm_attention.png")
        )
    else:
        # Basic attention heatmap
        fig = create_attention_heatmap(
            attention_weights=attention_weights,
            x_labels=encoder_tokens,
            y_labels=decoder_tokens,
            title="Attention Heatmap",
            save_path=os.path.join(args.output_dir, "attention_heatmap.png")
        )
    
    plt.close(fig)
    print(f"Visualization saved to {args.output_dir}")


if __name__ == "__main__":
    main() 