"""
Tokenization utilities for QA models.

This module provides helper functions for tokenization, including 
special token handling, sequence length management, and mask creation.
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union, Any


def get_tokenizer(model_name: str, add_special_qa_tokens: bool = True) -> AutoTokenizer:
    """
    Get a tokenizer with optional special QA tokens added.
    
    Args:
        model_name: Name of the pre-trained model or tokenizer.
        add_special_qa_tokens: Whether to add special tokens for QA tasks.
        
    Returns:
        Configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if add_special_qa_tokens:
        special_tokens = {
            'additional_special_tokens': ['[CTX]', '[QUES]', '[ANS]']
        }
        tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer


def get_special_token_ids(tokenizer: AutoTokenizer) -> Dict[str, int]:
    """
    Get the IDs of special tokens used in QA.
    
    Args:
        tokenizer: The tokenizer to use.
        
    Returns:
        Dictionary mapping token names to their IDs.
    """
    special_tokens = {
        "context": tokenizer.convert_tokens_to_ids('[CTX]'),
        "question": tokenizer.convert_tokens_to_ids('[QUES]'),
        "answer": tokenizer.convert_tokens_to_ids('[ANS]'),
        "pad": tokenizer.pad_token_id,
        "eos": tokenizer.eos_token_id,
        "bos": tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    }
    
    return special_tokens


def encode_qa_pair(
    context: str,
    question: str,
    tokenizer,
    add_special_tokens: bool = True,
    max_context_length: int = 384,
    max_question_length: int = 64,
    return_tensors: bool = False,
) -> Dict:
    """
    Encode a context-question pair for model input.
    
    Args:
        context: Context text.
        question: Question text.
        tokenizer: Tokenizer to use.
        add_special_tokens: Whether to add special tokens.
        max_context_length: Maximum context length.
        max_question_length: Maximum question length.
        return_tensors: Whether to return as tensors.
        
    Returns:
        Dictionary with encoder inputs.
    """
    # Tokenize context and question
    context_tokens = tokenizer.tokenize(context)
    question_tokens = tokenizer.tokenize(question)
    
    # Truncate if needed
    if len(context_tokens) > max_context_length:
        context_tokens = context_tokens[:max_context_length]
    
    if len(question_tokens) > max_question_length:
        question_tokens = question_tokens[:max_question_length]
    
    # Create encoder input
    encoder_input = []
    
    # Add context with special token if configured
    if add_special_tokens:
        encoder_input.append('[CTX]')
    encoder_input.extend(context_tokens)
    
    # Add question with special token if configured
    if add_special_tokens:
        encoder_input.append('[QUES]')
    encoder_input.extend(question_tokens)
    
    # Convert to IDs
    encoder_input_ids = tokenizer.convert_tokens_to_ids(encoder_input)
    
    # Create attention mask
    encoder_mask = [1] * len(encoder_input_ids)
    
    # Create result
    result = {
        "encoder_input": encoder_input_ids,
        "encoder_mask": encoder_mask
    }
    
    # Convert to tensors if requested
    if return_tensors:
        result["encoder_input"] = torch.tensor(encoder_input_ids, dtype=torch.long)
        result["encoder_mask"] = torch.tensor(encoder_mask, dtype=torch.long)
    
    return result


def create_attention_mask(
    sequence_length: int,
    valid_length: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Create an attention mask for a sequence.
    
    Args:
        sequence_length: Total length of the sequence (including padding).
        valid_length: Length of valid (non-padding) tokens.
        device: Device to create the tensor on.
        
    Returns:
        Attention mask tensor (1 for tokens, 0 for padding).
    """
    mask = torch.zeros(sequence_length, dtype=torch.long, device=device)
    mask[:valid_length] = 1
    return mask


def create_causal_mask(length: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Create a causal (triangular) mask for decoder self-attention.
    
    Args:
        length: Sequence length.
        device: Device to create the tensor on.
        
    Returns:
        Causal mask tensor (1 for allowed attention, 0 for masked).
    """
    mask = torch.tril(torch.ones(length, length, dtype=torch.long, device=device))
    return mask


def create_padding_mask(
    batch_size: int,
    sequence_length: int,
    valid_lengths: List[int],
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Create padding masks for a batch of sequences.
    
    Args:
        batch_size: Batch size.
        sequence_length: Maximum sequence length in the batch.
        valid_lengths: List of valid lengths for each sequence.
        device: Device to create the tensor on.
        
    Returns:
        Batch of attention masks (1 for tokens, 0 for padding).
    """
    masks = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=device)
    
    for i, length in enumerate(valid_lengths):
        masks[i, :length] = 1
    
    return masks


def decode_tokens(token_ids: List[int], tokenizer, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs to text.
    
    Args:
        token_ids: List of token IDs.
        tokenizer: Tokenizer to use.
        skip_special_tokens: Whether to skip special tokens.
        
    Returns:
        Decoded text.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def prepare_qa_inputs(
    contexts: List[str],
    questions: List[str],
    tokenizer,
    max_context_length: int = 384,
    max_question_length: int = 64,
    add_special_tokens: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for a batch of QA examples.
    
    Args:
        contexts: List of context texts.
        questions: List of question texts.
        tokenizer: Tokenizer to use.
        max_context_length: Maximum context length.
        max_question_length: Maximum question length.
        add_special_tokens: Whether to add special tokens.
        device: Device to put tensors on.
        
    Returns:
        Dictionary with model inputs.
    """
    # Encode each example
    encoded_inputs = [
        encode_qa_pair(
            context=context,
            question=question,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            max_context_length=max_context_length,
            max_question_length=max_question_length,
            return_tensors=False
        )
        for context, question in zip(contexts, questions)
    ]
    
    # Get max length for padding
    max_length = max(len(ex["encoder_input"]) for ex in encoded_inputs)
    
    # Initialize tensors
    encoder_inputs = []
    encoder_masks = []
    
    # Pad sequences
    for ex in encoded_inputs:
        encoder_input = ex["encoder_input"] + [tokenizer.pad_token_id] * (max_length - len(ex["encoder_input"]))
        encoder_mask = ex["encoder_mask"] + [0] * (max_length - len(ex["encoder_mask"]))
        
        encoder_inputs.append(encoder_input)
        encoder_masks.append(encoder_mask)
    
    # Convert to tensors
    encoder_inputs_tensor = torch.tensor(encoder_inputs, dtype=torch.long)
    encoder_masks_tensor = torch.tensor(encoder_masks, dtype=torch.long)
    
    # Move to device if specified
    if device is not None:
        encoder_inputs_tensor = encoder_inputs_tensor.to(device)
        encoder_masks_tensor = encoder_masks_tensor.to(device)
    
    return {
        "encoder_inputs": encoder_inputs_tensor,
        "encoder_mask": encoder_masks_tensor
    }
