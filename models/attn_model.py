"""
LSTM Encoder-Decoder model with Bahdanau attention for Question Answering.

This module implements a BiLSTM encoder and LSTM decoder with attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import random
import numpy as np
import math


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention mechanism."""
    
    def __init__(self, hidden_size: int, key_size: Optional[int] = None, query_size: Optional[int] = None):
        """
        Initialize Bahdanau attention.
        
        Args:
            hidden_size: Size of hidden states.
            key_size: Size of keys (encoder output size), defaults to hidden_size.
            query_size: Size of queries (decoder hidden size), defaults to hidden_size.
        """
        super(BahdanauAttention, self).__init__()
        
        # Set sizes
        key_size = key_size or hidden_size
        query_size = query_size or hidden_size
        
        # Define layers
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # For scaling attention scores
        self.scale = 1. / math.sqrt(hidden_size)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights for attention layers."""
        for layer in [self.key_layer, self.query_layer, self.energy_layer]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attention weights and context vector.
        
        Args:
            query: Decoder state [batch_size, query_size].
            encoder_outputs: Encoder outputs [batch_size, seq_len, key_size].
            mask: Mask for encoder outputs [batch_size, seq_len].
        
        Returns:
            Tuple of context vector and attention weights.
        """
        # Project query
        query = self.query_layer(query)  # [batch_size, hidden_size]
        
        # Project keys (encoder outputs)
        keys = self.key_layer(encoder_outputs)  # [batch_size, seq_len, hidden_size]
        
        # Calculate scores
        # Add query to all positions of encoder outputs and pass through energy layer
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        scores = self.energy_layer(torch.tanh(query + keys))  # [batch_size, seq_len, 1]
        scores = scores.squeeze(2)  # [batch_size, seq_len]
        
        # Scale scores
        scores = scores * self.scale
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]
        
        # Calculate context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, key_size]
        context = context.squeeze(1)  # [batch_size, key_size]
        
        return context, attention_weights


class AttentionLSTMDecoder(nn.Module):
    """LSTM Decoder with Bahdanau attention."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        encoder_hidden_size: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        """
        Initialize LSTM decoder with attention.
        
        Args:
            vocab_size: Size of vocabulary.
            embed_size: Size of embeddings.
            hidden_size: Size of decoder hidden states.
            encoder_hidden_size: Size of encoder outputs, defaults to hidden_size.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            padding_idx: Index used for padding.
        """
        super(AttentionLSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size or hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=padding_idx
        )
        
        # Attention mechanism
        self.attention = BahdanauAttention(
            hidden_size=hidden_size,
            key_size=self.encoder_hidden_size,
            query_size=hidden_size
        )
        
        # LSTM layer
        # Input is embedding + context vector
        self.lstm = nn.LSTM(
            input_size=embed_size + self.encoder_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        # Output layer
        # Combine LSTM output with context vector
        self.output_layer = nn.Linear(hidden_size + self.encoder_hidden_size, vocab_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through the decoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            hidden: Initial hidden state [num_layers, batch_size, hidden_size].
            cell: Initial cell state [num_layers, batch_size, hidden_size].
            encoder_outputs: Encoder outputs [batch_size, enc_seq_len, encoder_hidden_size].
            encoder_mask: Mask for encoder outputs [batch_size, enc_seq_len].
            
        Returns:
            Tuple of decoder outputs, final hidden states, and attention weights.
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Embed input tokens
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_size]
        embedded = self.dropout_layer(embedded)
        
        # Initialize outputs tensor
        outputs = torch.zeros(
            batch_size, seq_len, self.hidden_size + self.encoder_hidden_size,
            device=input_ids.device
        )
        
        # Initialize attention weights tensor
        attention_weights = torch.zeros(
            batch_size, seq_len, encoder_outputs.size(1),
            device=input_ids.device
        )
        
        # Process one token at a time
        for t in range(seq_len):
            # Get the hidden state to use for attention (top layer)
            # hidden shape: [num_layers, batch_size, hidden_size]
            query = hidden[-1]  # [batch_size, hidden_size]
            
            # Apply attention
            context, attn_weights = self.attention(
                query=query,
                encoder_outputs=encoder_outputs,
                mask=encoder_mask
            )
            
            # Save attention weights
            attention_weights[:, t] = attn_weights
            
            # Combine embedding with context vector
            lstm_input = torch.cat(
                [embedded[:, t:t+1], context.unsqueeze(1)], dim=2
            )  # [batch_size, 1, embed_size + encoder_hidden_size]
            
            # Pass through LSTM
            lstm_output, (hidden, cell) = self.lstm(
                lstm_input, (hidden, cell)
            )  # lstm_output: [batch_size, 1, hidden_size]
            
            # Combine LSTM output with context vector
            output = torch.cat(
                [lstm_output.squeeze(1), context], dim=1
            )  # [batch_size, hidden_size + encoder_hidden_size]
            
            # Save output
            outputs[:, t] = output
        
        # Apply output layer to get logits
        predictions = self.output_layer(self.dropout_layer(outputs))
        # predictions: [batch_size, seq_len, vocab_size]
        
        return predictions, (hidden, cell), attention_weights


class LSTMEncoderAttnDecoder(nn.Module):
    """LSTM Encoder-Decoder model with Bahdanau attention for QA."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        max_decode_length: int = 50,
        teacher_forcing_ratio: float = 0.5,
        save_attention: bool = True
    ):
        """
        Initialize LSTM Encoder-Decoder model with attention.
        
        Args:
            vocab_size: Size of vocabulary.
            embed_size: Size of embeddings.
            hidden_size: Size of hidden states.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            padding_idx: Index used for padding.
            sos_token_id: Start of sequence token ID.
            eos_token_id: End of sequence token ID.
            max_decode_length: Maximum length for decoding.
            teacher_forcing_ratio: Probability of using teacher forcing.
            save_attention: Whether to save attention weights during generation.
        """
        super(LSTMEncoderAttnDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.max_decode_length = max_decode_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.save_attention = save_attention
        
        # Reuse LSTMEncoder from lstm_model.py
        from models.lstm_model import LSTMEncoder
        
        # Initialize encoder
        self.encoder = LSTMEncoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx
        )
        
        # Initialize decoder with attention
        self.decoder = AttentionLSTMDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            encoder_hidden_size=hidden_size,  # Same size as encoder outputs
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx
        )
        
        # Store attention weights during generation (for visualization)
        self.stored_attention_weights = None
    
    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            encoder_inputs: Input token IDs for encoder [batch_size, enc_seq_len].
            decoder_inputs: Input token IDs for decoder [batch_size, dec_seq_len].
            encoder_mask: Attention mask for encoder [batch_size, enc_seq_len].
            decoder_mask: Attention mask for decoder [batch_size, dec_seq_len].
            targets: Target token IDs [batch_size, dec_seq_len].
        
        Returns:
            Dictionary with model outputs, including logits, loss, and attention weights.
        """
        batch_size = encoder_inputs.size(0)
        
        # Encoder forward pass
        encoder_outputs, (hidden, cell) = self.encoder(encoder_inputs, encoder_mask)
        
        # Dictionary to store outputs
        outputs = {"encoder_last_hidden_state": encoder_outputs}
        
        # Clear stored attention weights
        self.stored_attention_weights = None
        
        # For training (with decoder inputs)
        if decoder_inputs is not None:
            decoder_max_length = decoder_inputs.size(1)
            
            # Teacher forcing: use the ground-truth next token as input
            if random.random() < self.teacher_forcing_ratio:
                # Decoder forward pass with all inputs at once
                logits, (final_hidden, final_cell), attention_weights = self.decoder(
                    decoder_inputs, hidden, cell, encoder_outputs, encoder_mask
                )
                outputs["logits"] = logits
                outputs["attention_weights"] = attention_weights
            
            # No teacher forcing: use model's own predictions as the next input
            else:
                # Initialize tensor to store all decoder outputs
                logits = torch.zeros(
                    batch_size, decoder_max_length, self.vocab_size,
                    device=encoder_inputs.device
                )
                
                # Initialize tensor to store attention weights
                attention_weights = torch.zeros(
                    batch_size, decoder_max_length, encoder_outputs.size(1),
                    device=encoder_inputs.device
                )
                
                # First input is SOS token or first token from decoder_inputs
                decoder_input = decoder_inputs[:, 0].unsqueeze(1)
                
                for t in range(decoder_max_length):
                    # Forward pass for one step
                    step_logits, (hidden, cell), step_attn_weights = self.decoder(
                        decoder_input, hidden, cell, encoder_outputs, encoder_mask
                    )
                    
                    # Save outputs
                    logits[:, t:t+1] = step_logits
                    attention_weights[:, t:t+1] = step_attn_weights.unsqueeze(1)
                    
                    # Next input is the highest probability token
                    top_token = step_logits.argmax(dim=-1)
                    
                    # Break if all sequences have generated EOS
                    if (top_token == self.eos_token_id).all():
                        break
                    
                    # If not at the last step, update decoder_input
                    if t < decoder_max_length - 1:
                        # If the target for this step is available, use it (for mixed teacher forcing)
                        if decoder_inputs is not None and t + 1 < decoder_inputs.size(1):
                            decoder_input = decoder_inputs[:, t+1].unsqueeze(1)
                        else:
                            decoder_input = top_token
                
                outputs["logits"] = logits
                outputs["attention_weights"] = attention_weights
            
            # Calculate loss if targets provided
            if targets is not None:
                # Calculate loss (ignore padding)
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
                
                # Reshape logits and targets for loss calculation
                # logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
                # targets: [batch_size, seq_len] -> [batch_size * seq_len]
                loss = loss_fct(
                    outputs["logits"].contiguous().view(-1, self.vocab_size),
                    targets.contiguous().view(-1)
                )
                
                outputs["loss"] = loss
        
        return outputs
    
    def generate(
        self,
        encoder_inputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences using the decoder with attention.
        
        Args:
            encoder_inputs: Input token IDs for encoder [batch_size, enc_seq_len].
            encoder_mask: Attention mask for encoder [batch_size, enc_seq_len].
            max_length: Maximum generation length.
            min_length: Minimum generation length.
            do_sample: Whether to sample from the distribution.
            temperature: Sampling temperature.
            top_k: Number of highest probability tokens to keep for top-k sampling.
            eos_token_id: End of sequence token ID.
            
        Returns:
            Dictionary with generated token IDs and attention weights.
        """
        if max_length is None:
            max_length = self.max_decode_length
        
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        
        batch_size = encoder_inputs.size(0)
        device = encoder_inputs.device
        
        # Encoder forward pass
        encoder_outputs, (hidden, cell) = self.encoder(encoder_inputs, encoder_mask)
        
        # Store generated token IDs
        generated_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        
        # Store attention weights for visualization
        attention_weights = torch.zeros(
            batch_size, max_length, encoder_outputs.size(1),
            device=device
        ) if self.save_attention else None
        
        # Start with SOS token
        decoder_input = torch.full(
            (batch_size, 1), self.sos_token_id, dtype=torch.long, device=device
        )
        
        # Track which sequences are complete
        complete_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one at a time
        for t in range(max_length):
            # Get logits and attention weights for next token
            logits, (hidden, cell), attn_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, encoder_mask
            )  # logits: [batch_size, 1, vocab_size]
            
            # Save attention weights if needed
            if self.save_attention:
                attention_weights[:, t] = attn_weights
            
            # Mask logits for EOS token if we haven't reached min_length
            if t < min_length:
                logits[:, :, eos_token_id] = -float('inf')
            
            # Apply temperature scaling
            if do_sample and temperature > 0:
                logits = logits / temperature
            
            # Apply top-k sampling if specified
            if do_sample and top_k is not None:
                # Zero out logits below top-k
                # First, get top-k values and their indices
                top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
                
                # Create a new logits tensor filled with -inf
                new_logits = torch.full_like(logits, -float('inf'))
                
                # Fill in the top-k logits
                for i in range(batch_size):
                    new_logits[i, 0, top_k_indices[i, 0]] = top_k_logits[i, 0]
                
                logits = new_logits
            
            # Get next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs[:, 0], 1)
            else:
                next_token = torch.argmax(logits, dim=-1)  # [batch_size, 1]
            
            # Save generated token
            generated_ids[:, t] = next_token.squeeze(1)
            
            # Update complete_sequences mask
            complete_sequences = complete_sequences | (next_token.squeeze(1) == eos_token_id)
            
            # Break if all sequences are complete
            if complete_sequences.all().item():
                break
            
            # Next input is the generated token
            decoder_input = next_token
        
        # Save attention weights for later visualization
        if self.save_attention:
            self.stored_attention_weights = attention_weights
            return {"generated_ids": generated_ids, "attention_weights": attention_weights}
        else:
            return {"generated_ids": generated_ids}
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get stored attention weights from the last generation.
        
        Returns:
            Attention weights tensor if available.
        """
        return self.stored_attention_weights
