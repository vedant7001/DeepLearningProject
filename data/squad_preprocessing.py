"""
SQuAD preprocessing module for Question Answering.

This module handles loading, processing, and preparing SQuAD dataset for
encoder-decoder models. It includes functions for tokenization, sequence
preparation, and batching.
"""

import os
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class SQuADDataset(Dataset):
    """SQuAD dataset for Question Answering task."""
    
    def __init__(
        self,
        data_split: str,
        tokenizer_name: str,
        max_context_length: int = 384,
        max_question_length: int = 64,
        max_answer_length: int = 50,
        use_v2: bool = False,
        add_special_tokens: bool = True,
    ):
        """
        Initialize SQuAD dataset.
        
        Args:
            data_split: Which split to use ("train", "val", or "test").
            tokenizer_name: Name of the tokenizer to use.
            max_context_length: Maximum context length.
            max_question_length: Maximum question length.
            max_answer_length: Maximum answer length.
            use_v2: Whether to use SQuAD v2.0 instead of v1.1.
            add_special_tokens: Whether to add special tokens for QA.
        """
        self.data_split = "validation" if data_split == "val" else data_split
        self.tokenizer_name = tokenizer_name
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.use_v2 = use_v2
        self.add_special_tokens = add_special_tokens
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add special tokens if needed
        if add_special_tokens:
            special_tokens = {
                'additional_special_tokens': ['[CTX]', '[QUES]', '[ANS]']
            }
            self.tokenizer.add_special_tokens(special_tokens)
        
        # Load dataset
        dataset_name = "squad_v2" if use_v2 else "squad"
        logger.info(f"Loading {dataset_name} dataset ({data_split} split)...")
        self.dataset = load_dataset(dataset_name)[self.data_split]
        
        # Process dataset
        self.processed_data = self._process_dataset()
        
        logger.info(f"Loaded {len(self.processed_data)} examples")
    
    def _process_dataset(self) -> List[Dict]:
        """
        Process the dataset for encoder-decoder format.
        
        Returns:
            List of processed examples.
        """
        processed_data = []
        
        for example in self.dataset:
            # Skip examples without answers (for SQuAD v2)
            if self.use_v2 and len(example["answers"]["text"]) == 0:
                continue
            
            # Get context, question, and answer
            context = example["context"]
            question = example["question"]
            
            # Take the first answer (SQuAD has multiple answers for the same question)
            answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
            
            # Skip examples with empty answers
            if not answer:
                continue
            
            # Tokenize context, question, and answer
            context_tokens = self.tokenizer.tokenize(context)
            question_tokens = self.tokenizer.tokenize(question)
            answer_tokens = self.tokenizer.tokenize(answer)
            
            # Truncate sequences if needed
            if len(context_tokens) > self.max_context_length:
                context_tokens = context_tokens[:self.max_context_length]
            
            if len(question_tokens) > self.max_question_length:
                question_tokens = question_tokens[:self.max_question_length]
            
            if len(answer_tokens) > self.max_answer_length:
                answer_tokens = answer_tokens[:self.max_answer_length]
            
            # Create encoder input
            encoder_input = []
            
            # Add context with special token if configured
            if self.add_special_tokens:
                encoder_input.append("[CTX]")
            encoder_input.extend(context_tokens)
            
            # Add question with special token if configured
            if self.add_special_tokens:
                encoder_input.append("[QUES]")
            encoder_input.extend(question_tokens)
            
            # Create decoder input and target
            decoder_input = []
            target = []
            
            # Add answer with special token if configured
            if self.add_special_tokens:
                decoder_input.append("[ANS]")
            
            # Add start token for decoder input (teacher forcing)
            if self.tokenizer.bos_token:
                decoder_input.append(self.tokenizer.bos_token)
            
            # Add answer tokens
            decoder_input.extend(answer_tokens)
            
            # Add end token
            if self.tokenizer.eos_token:
                decoder_input.append(self.tokenizer.eos_token)
            
            # Target is decoder input shifted right (next token prediction)
            # For the target, we start with the answer tokens (no BOS)
            target.extend(answer_tokens)
            
            # Add end token to target
            if self.tokenizer.eos_token:
                target.append(self.tokenizer.eos_token)
            
            # Convert tokens to IDs
            encoder_input_ids = self.tokenizer.convert_tokens_to_ids(encoder_input)
            decoder_input_ids = self.tokenizer.convert_tokens_to_ids(decoder_input)
            target_ids = self.tokenizer.convert_tokens_to_ids(target)
            
            # Create attention masks (1 for tokens, 0 for padding)
            encoder_mask = [1] * len(encoder_input_ids)
            decoder_mask = [1] * len(decoder_input_ids)
            
            # Add example to processed data
            processed_data.append({
                "id": example["id"],
                "context": context,
                "question": question,
                "answer": answer,
                "encoder_input_ids": encoder_input_ids,
                "decoder_input_ids": decoder_input_ids,
                "target_ids": target_ids,
                "encoder_mask": encoder_mask,
                "decoder_mask": decoder_mask,
            })
        
        return processed_data
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get example by index."""
        return self.processed_data[idx]


def collate_squad_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch of examples with padding.
    
    Args:
        batch: List of examples from SQuADDataset.
        
    Returns:
        Dict with tensors for model input.
    """
    # Get max lengths in this batch
    max_encoder_len = max(len(ex["encoder_input_ids"]) for ex in batch)
    max_decoder_len = max(len(ex["decoder_input_ids"]) for ex in batch)
    
    # Initialize tensors for padding
    encoder_inputs = []
    decoder_inputs = []
    targets = []
    encoder_masks = []
    decoder_masks = []
    ids = []
    contexts = []
    questions = []
    answers = []
    
    # Pad sequences and create tensors
    for ex in batch:
        # Pad encoder input
        encoder_input = ex["encoder_input_ids"] + [0] * (max_encoder_len - len(ex["encoder_input_ids"]))
        encoder_inputs.append(encoder_input)
        
        # Pad encoder mask
        encoder_mask = ex["encoder_mask"] + [0] * (max_encoder_len - len(ex["encoder_mask"]))
        encoder_masks.append(encoder_mask)
        
        # Pad decoder input
        decoder_input = ex["decoder_input_ids"] + [0] * (max_decoder_len - len(ex["decoder_input_ids"]))
        decoder_inputs.append(decoder_input)
        
        # Pad target
        target = ex["target_ids"] + [0] * (max_decoder_len - len(ex["target_ids"]))
        targets.append(target)
        
        # Pad decoder mask
        decoder_mask = ex["decoder_mask"] + [0] * (max_decoder_len - len(ex["decoder_mask"]))
        decoder_masks.append(decoder_mask)
        
        # Add metadata
        ids.append(ex["id"])
        contexts.append(ex["context"])
        questions.append(ex["question"])
        answers.append(ex["answer"])
    
    # Convert to tensors
    return {
        "ids": ids,
        "contexts": contexts,
        "questions": questions,
        "answers": answers,
        "encoder_inputs": torch.tensor(encoder_inputs, dtype=torch.long),
        "decoder_inputs": torch.tensor(decoder_inputs, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "encoder_mask": torch.tensor(encoder_masks, dtype=torch.long),
        "decoder_mask": torch.tensor(decoder_masks, dtype=torch.long),
    }


def get_squad_dataloader(
    data_split: str,
    tokenizer_name: str,
    batch_size: int = 16,
    shuffle: bool = True,
    max_context_length: int = 384,
    max_question_length: int = 64,
    max_answer_length: int = 50,
    use_v2: bool = False,
    add_special_tokens: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Get a DataLoader for SQuAD dataset.
    
    Args:
        data_split: Which split to use ("train", "val", or "test").
        tokenizer_name: Name of the tokenizer to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.
        max_context_length: Maximum context length.
        max_question_length: Maximum question length.
        max_answer_length: Maximum answer length.
        use_v2: Whether to use SQuAD v2.0 instead of v1.1.
        add_special_tokens: Whether to add special tokens for QA.
        num_workers: Number of worker processes for data loading.
        
    Returns:
        DataLoader for SQuAD dataset.
    """
    # Create dataset
    dataset = SQuADDataset(
        data_split=data_split,
        tokenizer_name=tokenizer_name,
        max_context_length=max_context_length,
        max_question_length=max_question_length,
        max_answer_length=max_answer_length,
        use_v2=use_v2,
        add_special_tokens=add_special_tokens
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_squad_batch,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


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
        encoder_input.append("[CTX]")
    encoder_input.extend(context_tokens)
    
    # Add question with special token if configured
    if add_special_tokens:
        encoder_input.append("[QUES]")
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


if __name__ == "__main__":
    # Test dataset and dataloader
    tokenizer_name = "bert-base-uncased"
    
    # Create a small test dataloader
    dataloader = get_squad_dataloader(
        data_split="train",
        tokenizer_name=tokenizer_name,
        batch_size=4,
        shuffle=True,
        use_v2=False
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Print batch information
    print("Batch information:")
    print(f"Encoder inputs shape: {batch['encoder_inputs'].shape}")
    print(f"Decoder inputs shape: {batch['decoder_inputs'].shape}")
    print(f"Targets shape: {batch['targets'].shape}")
    print(f"Encoder mask shape: {batch['encoder_mask'].shape}")
    print(f"Decoder mask shape: {batch['decoder_mask'].shape}")
    
    # Print an example
    print("\nExample:")
    idx = 0
    print(f"Context: {batch['contexts'][idx]}")
    print(f"Question: {batch['questions'][idx]}")
    print(f"Answer: {batch['answers'][idx]}")
    
    # Load tokenizer to decode tokens
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special_tokens = {
        'additional_special_tokens': ['[CTX]', '[QUES]', '[ANS]']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    print(f"Encoder input: {tokenizer.decode(batch['encoder_inputs'][idx].tolist(), skip_special_tokens=False)}")
    print(f"Decoder input: {tokenizer.decode(batch['decoder_inputs'][idx].tolist(), skip_special_tokens=False)}")
    print(f"Target: {tokenizer.decode(batch['targets'][idx].tolist(), skip_special_tokens=False)}")
