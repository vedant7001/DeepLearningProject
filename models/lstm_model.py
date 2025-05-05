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


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM Encoder."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        """
        Initialize BiLSTM encoder.
        
        Args:
            vocab_size: Size of vocabulary.
            embed_size: Size of embeddings.
            hidden_size: Size of hidden states.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            padding_idx: Index used for padding.
        """
        super(LSTMEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=padding_idx
        )
        
        # BiLSTM Encoder
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Linear layer to combine bidirectional outputs
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            mask: Attention mask [batch_size, seq_len].
            
        Returns:
            Tuple of encoder outputs and final hidden states.
        """
        # Embed input tokens
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_size]
        embedded = self.dropout_layer(embedded)
        
        # Create packed sequence if mask is provided
        if mask is not None:
            # Calculate sequence lengths from mask
            seq_lengths = mask.sum(dim=1).cpu()
            
            # Sort sequences by length in descending order
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            embedded = embedded[perm_idx]
            
            # Create packed sequence
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, seq_lengths, batch_first=True
            )
            
            # Pass through BiLSTM
            packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
            
            # Unpack outputs
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
            
            # Restore original order
            _, unperm_idx = perm_idx.sort(0)
            outputs = outputs[unperm_idx]
            hidden = hidden[:, unperm_idx]
            cell = cell[:, unperm_idx]
        else:
            # Without mask, just pass through BiLSTM
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional outputs
        # outputs shape: [batch_size, seq_len, 2*hidden_size]
        outputs = self.linear(outputs)  # [batch_size, seq_len, hidden_size]
        
        # Process hidden state for decoder initialization
        # Combine forward and backward hidden states
        # hidden shape: [num_layers*2, batch_size, hidden_size]
        
        # Reshape to separate directions: [num_layers, 2, batch_size, hidden_size]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        
        # Combine directions: [num_layers, batch_size, hidden_size]
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
        
        # Project to decoder hidden size
        hidden = self.linear(hidden)
        cell = self.linear(cell)
        
        return outputs, (hidden, cell)


class LSTMDecoder(nn.Module):
    """LSTM Decoder for sequence generation."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        """
        Initialize LSTM decoder.
        
        Args:
            vocab_size: Size of vocabulary.
            embed_size: Size of embeddings.
            hidden_size: Size of hidden states.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            padding_idx: Index used for padding.
        """
        super(LSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=padding_idx
        )
        
        # LSTM layer (unidirectional)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the decoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            hidden: Initial hidden state [num_layers, batch_size, hidden_size].
            cell: Initial cell state [num_layers, batch_size, hidden_size].
            
        Returns:
            Tuple of decoder outputs and final hidden states.
        """
        # Embed input tokens
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_size]
        embedded = self.dropout_layer(embedded)
        
        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(
            embedded, (hidden, cell)
        )  # outputs: [batch_size, seq_len, hidden_size]
        
        # Apply output layer to get logits
        predictions = self.output_layer(self.dropout_layer(outputs))
        # predictions: [batch_size, seq_len, vocab_size]
        
        return predictions, (hidden, cell)


class LSTMEncoderDecoder(nn.Module):
    """LSTM Encoder-Decoder model for QA."""
    
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
        teacher_forcing_ratio: float = 0.5
    ):
        """
        Initialize LSTM Encoder-Decoder model.
        
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
        """
        super(LSTMEncoderDecoder, self).__init__()
        
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
        
        # Initialize encoder and decoder
        self.encoder = LSTMEncoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx
        )
        
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx
        )
    
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
            Dictionary with model outputs, including logits and loss if targets provided.
        """
        batch_size = encoder_inputs.size(0)
        
        # Encoder forward pass
        encoder_outputs, (hidden, cell) = self.encoder(encoder_inputs, encoder_mask)
        
        # Dictionary to store outputs
        outputs = {"encoder_last_hidden_state": encoder_outputs}
        
        # For training (with decoder inputs)
        if decoder_inputs is not None:
            decoder_max_length = decoder_inputs.size(1)
            
            # Teacher forcing: use the ground-truth next token as input
            if random.random() < self.teacher_forcing_ratio:
                # Decoder forward pass with all inputs at once
                logits, (final_hidden, final_cell) = self.decoder(
                    decoder_inputs, hidden, cell
                )
                outputs["logits"] = logits
            
            # No teacher forcing: use model's own predictions as the next input
            else:
                # Initialize tensor to store all decoder outputs
                logits = torch.zeros(
                    batch_size, decoder_max_length, self.vocab_size,
                    device=encoder_inputs.device
                )
                
                # First input is SOS token or first token from decoder_inputs
                decoder_input = decoder_inputs[:, 0].unsqueeze(1)
                
                for t in range(decoder_max_length):
                    step_logits, (hidden, cell) = self.decoder(
                        decoder_input, hidden, cell
                    )
                    
                    # Save current step logits
                    logits[:, t:t+1] = step_logits
                    
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
            
            # Calculate loss if targets provided
            if targets is not None:
                # Shift targets for calculating loss (we predict the next token)
                shifted_targets = targets
                
                # Calculate loss (ignore padding)
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
                
                # Reshape logits and targets for loss calculation
                # logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
                # targets: [batch_size, seq_len] -> [batch_size * seq_len]
                loss = loss_fct(
                    outputs["logits"].contiguous().view(-1, self.vocab_size),
                    shifted_targets.contiguous().view(-1)
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
        Generate sequences using the decoder.
        
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
            Dictionary with generated token IDs.
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
        
        # Start with SOS token
        decoder_input = torch.full(
            (batch_size, 1), self.sos_token_id, dtype=torch.long, device=device
        )
        
        # Track which sequences are complete
        complete_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one at a time
        for t in range(max_length):
            # Get logits for next token
            logits, (hidden, cell) = self.decoder(
                decoder_input, hidden, cell
            )  # logits: [batch_size, 1, vocab_size]
            
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
        
        return {"generated_ids": generated_ids}
