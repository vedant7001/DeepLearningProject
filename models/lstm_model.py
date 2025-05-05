"""
LSTM Encoder-Decoder model for Question Answering.

This module implements a BiLSTM encoder and LSTM decoder without attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import random
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        
        # Get embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_size]
        embedded = self.dropout_layer(embedded)
        
        # Instead of packing the sequence, we'll run the LSTM directly
        # and then mask the outputs using the attention mask
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, (hidden, cell)


class LSTMEncoderDecoder(nn.Module):
    """
    LSTM Encoder-Decoder model for Question Answering.
    This class is a wrapper around the QAModel for compatibility with the training module.
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        embed_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        sos_token_id: int = 101,
        eos_token_id: int = 102
    ):
        super().__init__()
        
        self.qa_model = QAModel(
            model_type="lstm",
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Store token IDs for reference
        self.padding_idx = padding_idx
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model"""
        return self.qa_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tokenizer = None
    ) -> Dict[str, torch.Tensor]:
        """Generate predictions using the model"""
        return self.qa_model.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer
        )


class QAModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        vocab_size: int = 30522,  # Default for bert-base-uncased
        embed_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Output layers for start and end positions
        lstm_hidden_size = hidden_size * 2  # bidirectional
        self.start_output = nn.Linear(lstm_hidden_size, 1)
        self.end_output = nn.Linear(lstm_hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get encoder outputs
        encoder_outputs, _ = self.encoder(input_ids, attention_mask)
        encoder_outputs = self.dropout(encoder_outputs)
        
        # Get start and end logits
        start_logits = self.start_output(encoder_outputs).squeeze(-1)  # [batch_size, seq_len]
        end_logits = self.end_output(encoder_outputs).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(~attention_mask.bool(), float('-inf'))
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float('-inf'))
        
        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits
        }
        
        # Calculate loss if training
        if start_positions is not None and end_positions is not None:
            # Clamp the target positions to be within the valid range
            # This prevents "Target X is out of bounds" errors
            seq_length = start_logits.size(1)
            start_positions = torch.clamp(start_positions, 0, seq_length - 1)
            end_positions = torch.clamp(end_positions, 0, seq_length - 1)
            
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            outputs['loss'] = (start_loss + end_loss) / 2
        
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tokenizer = None
    ) -> Dict[str, torch.Tensor]:
        # Get predictions
        outputs = self.forward(input_ids, attention_mask)
        
        # Get start and end positions
        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']
        
        # Get the most likely start and end positions
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)
        
        # Get top k start and end positions
        k = 5
        start_top_k = torch.topk(start_probs, k, dim=-1)
        end_top_k = torch.topk(end_probs, k, dim=-1)
        
        # Find best start-end pair
        max_answer_length = 30
        predictions = []
        
        for i in range(start_logits.size(0)):
            valid_answers = []
            
            for start_idx in start_top_k.indices[i]:
                for end_idx in end_top_k.indices[i]:
                    if end_idx < start_idx:
                        continue
                    if end_idx - start_idx + 1 > max_answer_length:
                        continue
                        
                    valid_answers.append({
                        'start': start_idx.item(),
                        'end': end_idx.item(),
                        'score': start_probs[i][start_idx] * end_probs[i][end_idx]
                    })
            
            if not valid_answers:
                predictions.append("")
                continue
                
            # Sort valid answers by score
            valid_answers = sorted(valid_answers, key=lambda x: x['score'], reverse=True)
            best_answer = valid_answers[0]
            
            # Convert to text if tokenizer provided
            if tokenizer:
                answer_tokens = input_ids[i][best_answer['start']:best_answer['end']+1]
                answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                predictions.append(answer_text)
            else:
                predictions.append((best_answer['start'], best_answer['end']))
        
        return {'predictions': predictions}
