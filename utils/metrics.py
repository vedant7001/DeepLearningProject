"""
Metrics utilities for evaluating Question Answering models.

This module includes implementations of common evaluation metrics
for QA systems including F1, BLEU, ROUGE-L, and exact match.
"""

import re
import string
import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter
from rouge_score import rouge_scorer
import sacrebleu
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download required NLTK packages if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for consistent evaluation.
    
    Args:
        s: Input string.
        
    Returns:
        Normalized string.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """
    Get tokens from a string.
    
    Args:
        s: Input string.
        
    Returns:
        List of tokens.
    """
    if not s:
        return []
    
    # Normalize
    s = normalize_answer(s)
    
    # Tokenize
    return s.split()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score.
    
    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.
        
    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    # Normalize
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    
    # Check exact match
    return float(prediction == ground_truth)


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score between prediction and ground truth.
    
    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.
        
    Returns:
        F1 score.
    """
    # Get tokens
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    
    # Empty answers
    if not prediction_tokens or not ground_truth_tokens:
        return int(prediction_tokens == ground_truth_tokens)
    
    # Count common tokens
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common.values())
    
    # Edge case
    if num_common == 0:
        return 0.0
    
    # Compute precision and recall
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)
    
    # Compute F1
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_bleu(prediction: str, ground_truth: str) -> float:
    """
    Compute BLEU score.
    
    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.
        
    Returns:
        BLEU score.
    """
    # Tokenize
    prediction_tokens = nltk.word_tokenize(prediction.lower())
    ground_truth_tokens = nltk.word_tokenize(ground_truth.lower())
    
    # Edge cases
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    
    # Use smoothing for short sentences
    smoothie = SmoothingFunction().method1
    
    # Compute BLEU (up to 4-grams or max possible)
    max_n = min(4, len(ground_truth_tokens))
    weights = [1.0/max_n] * max_n
    
    try:
        bleu = sentence_bleu([ground_truth_tokens], prediction_tokens, 
                             weights=weights, smoothing_function=smoothie)
        return bleu
    except Exception:
        return 0.0


def compute_rouge(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    
    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.
        
    Returns:
        Dictionary with ROUGE scores.
    """
    # Create scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Compute scores
    scores = scorer.score(ground_truth, prediction)
    
    # Extract F1 scores
    rouge_scores = {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }
    
    return rouge_scores


def compute_metrics(
    predictions: List[str],
    ground_truths: List[str],
    include_rouge: bool = True,
    include_bleu: bool = True
) -> Dict[str, float]:
    """
    Compute multiple metrics for a list of predictions.
    
    Args:
        predictions: List of predicted answers.
        ground_truths: List of ground truth answers.
        include_rouge: Whether to include ROUGE scores.
        include_bleu: Whether to include BLEU score.
        
    Returns:
        Dictionary with metrics.
    """
    # Ensure predictions and ground truths have the same length
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have the same length"
    
    # Initialize metrics
    exact_match = 0.0
    f1 = 0.0
    bleu = 0.0
    rouge1_f = 0.0
    rouge2_f = 0.0
    rougeL_f = 0.0
    
    # Compute metrics for each example
    for pred, gt in zip(predictions, ground_truths):
        # Exact match and F1
        exact_match += compute_exact_match(pred, gt)
        f1 += compute_f1(pred, gt)
        
        # BLEU
        if include_bleu:
            bleu += compute_bleu(pred, gt)
        
        # ROUGE
        if include_rouge:
            rouge_scores = compute_rouge(pred, gt)
            rouge1_f += rouge_scores['rouge1_f']
            rouge2_f += rouge_scores['rouge2_f']
            rougeL_f += rouge_scores['rougeL_f']
    
    # Normalize by number of examples
    n = len(predictions)
    metrics = {
        'exact_match': exact_match / n * 100,  # Convert to percentage
        'f1': f1 / n * 100  # Convert to percentage
    }
    
    # Add BLEU
    if include_bleu:
        metrics['bleu'] = bleu / n * 100  # Convert to percentage
    
    # Add ROUGE
    if include_rouge:
        metrics['rouge1_f'] = rouge1_f / n * 100  # Convert to percentage
        metrics['rouge2_f'] = rouge2_f / n * 100  # Convert to percentage
        metrics['rougeL_f'] = rougeL_f / n * 100  # Convert to percentage
    
    return metrics
