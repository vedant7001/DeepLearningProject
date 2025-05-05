"""
Transformer Encoder-Decoder model for Question Answering.

This module implements a Transformer model following the 
"Attention Is All You Need" architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model."""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Hidden dimension size.
            max_seq_length: Maximum sequence length.
            dropout: Dropout probability.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model].
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerQA(nn.Module):
    """Transformer Encoder-Decoder model for Question Answering."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        padding_idx: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        use_native_transformer: bool = True,
        save_attention: bool = True
    ):
        """
        Initialize transformer model.
        
        Args:
            vocab_size: Size of vocabulary.
            d_model: Hidden dimension size.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout probability.
            max_seq_length: Maximum sequence length.
            padding_idx: Index used for padding.
            sos_token_id: Start of sequence token ID.
            eos_token_id: End of sequence token ID.
            use_native_transformer: Whether to use PyTorch's nn.Transformer.
            save_attention: Whether to save attention weights.
        """
        super(TransformerQA, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.use_native_transformer = use_native_transformer
        self.save_attention = save_attention
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Transformer model
        if use_native_transformer:
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
        else:
            # Custom transformer components
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_encoder_layers,
                norm=nn.LayerNorm(d_model)
            )
            
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_decoder_layers,
                norm=nn.LayerNorm(d_model)
            )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Scaling factor for embeddings
        self.scale = math.sqrt(d_model)
        
        # For storing attention weights
        self.stored_attentions = None
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate a square causal mask for the decoder.
        
        Args:
            sz: Size of the square mask.
            device: Device to create the mask on.
            
        Returns:
            Causal mask where elements in the upper triangle are -inf.
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _make_source_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for source sequence.
        
        Args:
            src: Source sequence [batch_size, src_len].
            
        Returns:
            Padding mask (1 for padding, 0 for non-padding).
        """
        src_mask = (src == self.padding_idx)
        return src_mask
    
    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the transformer model.
        
        Args:
            encoder_inputs: Input token IDs for encoder [batch_size, enc_seq_len].
            decoder_inputs: Input token IDs for decoder [batch_size, dec_seq_len].
            encoder_mask: Attention mask for encoder [batch_size, enc_seq_len].
            decoder_mask: Attention mask for decoder [batch_size, dec_seq_len].
            targets: Target token IDs [batch_size, dec_seq_len].
            
        Returns:
            Dictionary with model outputs.
        """
        # Get device
        device = encoder_inputs.device
        
        # Get sequence lengths
        src_seq_len = encoder_inputs.size(1)
        
        # Create source mask if not provided
        if encoder_mask is None:
            src_key_padding_mask = self._make_source_mask(encoder_inputs)
        else:
            # Convert from attention mask (1=attend, 0=ignore) to key padding mask (True=ignore, False=attend)
            src_key_padding_mask = (encoder_mask == 0)
        
        # Embed inputs
        src_embedded = self.embedding(encoder_inputs) * self.scale
        src_embedded = self.positional_encoding(src_embedded)
        
        # Initialize outputs dictionary
        outputs = {}
        
        # If we have decoder inputs, run the full encoder-decoder model
        if decoder_inputs is not None:
            tgt_seq_len = decoder_inputs.size(1)
            
            # Create target mask if not provided
            if decoder_mask is None:
                tgt_key_padding_mask = self._make_source_mask(decoder_inputs)
            else:
                # Convert from attention mask (1=attend, 0=ignore) to key padding mask (True=ignore, False=attend)
                tgt_key_padding_mask = (decoder_mask == 0)
            
            # Create causal mask for the decoder
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)
            
            # Embed decoder inputs
            tgt_embedded = self.embedding(decoder_inputs) * self.scale
            tgt_embedded = self.positional_encoding(tgt_embedded)
            
            # Forward pass through transformer
            if self.use_native_transformer:
                # PyTorch's Transformer expects different mask formats
                memory_key_padding_mask = src_key_padding_mask
                
                output = self.transformer(
                    src=src_embedded,
                    tgt=tgt_embedded,
                    src_mask=None,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
            else:
                # Custom transformer components
                # Encoder
                memory = self.transformer_encoder(
                    src_embedded,
                    src_key_padding_mask=src_key_padding_mask
                )
                
                # Decoder
                output = self.transformer_decoder(
                    tgt_embedded,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
            
            # Apply output layer to get logits
            logits = self.output_layer(output)
            outputs["logits"] = logits
            
            # Calculate loss if targets provided
            if targets is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
                loss = loss_fct(
                    logits.contiguous().view(-1, self.vocab_size),
                    targets.contiguous().view(-1)
                )
                outputs["loss"] = loss
        else:
            # Run encoder only
            if self.use_native_transformer:
                # We need to use a dummy decoder input for nn.Transformer
                dummy_tgt = torch.zeros(
                    (encoder_inputs.size(0), 1),
                    dtype=torch.long,
                    device=device
                ).fill_(self.sos_token_id)
                dummy_tgt_embedded = self.embedding(dummy_tgt) * self.scale
                dummy_tgt_embedded = self.positional_encoding(dummy_tgt_embedded)
                
                # Forward pass through encoder only
                # We'll discard the decoder output
                memory = self.transformer.encoder(
                    src=src_embedded,
                    mask=None,
                    src_key_padding_mask=src_key_padding_mask
                )
            else:
                # Custom transformer encoder
                memory = self.transformer_encoder(
                    src_embedded,
                    src_key_padding_mask=src_key_padding_mask
                )
            
            outputs["encoder_output"] = memory
        
        return outputs
    
    def generate(
        self,
        encoder_inputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
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
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        
        batch_size = encoder_inputs.size(0)
        device = encoder_inputs.device
        
        # Get source mask
        if encoder_mask is None:
            src_key_padding_mask = self._make_source_mask(encoder_inputs)
        else:
            # Convert from attention mask (1=attend, 0=ignore) to key padding mask (True=ignore, False=attend)
            src_key_padding_mask = (encoder_mask == 0)
        
        # Embed inputs
        src_embedded = self.embedding(encoder_inputs) * self.scale
        src_embedded = self.positional_encoding(src_embedded)
        
        # Encode source sequence
        if self.use_native_transformer:
            memory = self.transformer.encoder(
                src=src_embedded,
                mask=None,
                src_key_padding_mask=src_key_padding_mask
            )
        else:
            memory = self.transformer_encoder(
                src_embedded,
                src_key_padding_mask=src_key_padding_mask
            )
        
        # Initialize decoder input with SOS token
        decoder_input = torch.full(
            (batch_size, 1),
            self.sos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Store generated tokens
        generated_ids = torch.zeros(
            batch_size, max_length,
            dtype=torch.long,
            device=device
        )
        
        # Store attention weights if needed
        if self.save_attention:
            # For now, just store the decoder self-attention
            # In a real implementation, we might also store encoder attention and 
            # encoder-decoder cross attention
            self.stored_attentions = {
                "decoder_self_attn": [],
                "encoder_decoder_attn": []
            }
        
        # Track which sequences are complete
        complete_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens auto-regressively
        for i in range(max_length):
            # Create causal mask for decoder
            # This ensures that positions in the decoder only attend to earlier positions
            tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1), device)
            
            # Get target padding mask
            tgt_key_padding_mask = self._make_source_mask(decoder_input)
            
            # Embed decoder inputs
            tgt_embedded = self.embedding(decoder_input) * self.scale
            tgt_embedded = self.positional_encoding(tgt_embedded)
            
            # Decode
            if self.use_native_transformer:
                # Using PyTorch's Transformer
                # Note: We can't easily extract attention weights from nn.Transformer
                output = self.transformer.decoder(
                    tgt=tgt_embedded,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
            else:
                # Using custom transformer decoder
                # When using custom layers, we could extract attention weights
                output = self.transformer_decoder(
                    tgt_embedded,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
            
            # Get predictions for next token
            logits = self.output_layer(output[:, -1])  # [batch_size, vocab_size]
            
            # Mask logits for EOS token if we haven't reached min_length
            if i < min_length:
                logits[:, eos_token_id] = -float('inf')
            
            # Apply temperature scaling
            if do_sample and temperature > 0:
                logits = logits / temperature
            
            # Apply top-k sampling if specified
            if do_sample and top_k is not None:
                # Zero out logits below top-k
                top_k_values, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
                logits_mask = torch.zeros_like(logits, dtype=torch.bool)
                logits_mask.scatter_(1, top_k_indices, True)
                logits = logits.masked_fill(~logits_mask, -float('inf'))
            
            # Sample or take argmax to get next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # [batch_size, 1]
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # Save generated token
            generated_ids[:, i] = next_token.squeeze(1)
            
            # Update complete_sequences mask
            complete_sequences = complete_sequences | (next_token.squeeze(1) == eos_token_id)
            
            # Break if all sequences are complete
            if complete_sequences.all().item():
                break
            
            # Update decoder input for next iteration
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        return {"generated_ids": generated_ids}
    
    def get_attention_weights(self) -> Dict[str, List[torch.Tensor]]:
        """
        Get stored attention weights from the last generation.
        
        Returns:
            Dictionary of attention weight tensors.
        """
        return self.stored_attentions


# Define a simple factory function to create the appropriate model type
def create_transformer_qa(
    vocab_size: int,
    model_size: str = "base",
    **kwargs
) -> TransformerQA:
    """
    Create a Transformer model based on the specified size.
    
    Args:
        vocab_size: Size of vocabulary.
        model_size: Model size ("small", "base", "large").
        **kwargs: Additional arguments to pass to TransformerQA.
    
    Returns:
        TransformerQA model.
    """
    model_configs = {
        "small": {
            "d_model": 256,
            "nhead": 4,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dim_feedforward": 1024
        },
        "base": {
            "d_model": 512,
            "nhead": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 2048
        },
        "large": {
            "d_model": 768,
            "nhead": 12,
            "num_encoder_layers": 12,
            "num_decoder_layers": 12,
            "dim_feedforward": 3072
        }
    }
    
    if model_size not in model_configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config = model_configs[model_size]
    
    # Override with any kwargs
    config.update(kwargs)
    
    # Add vocab_size
    config["vocab_size"] = vocab_size
    
    return TransformerQA(**config)
