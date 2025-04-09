"""
Module: Induced Point Featurizer with Cross-Attention

This module implements a neural network architecture for feature extraction
using a cross-attention mechanism. The architecture combines learnable seed
vectors and input data through iterative attention rounds to encode meaningful
representations.

Classes:
--------
1. CrossAttentionBlock:
    - Implements a single block of cross-attention with residual connections
      and a feed-forward network.

2. Encoder:
    - Alternates cross-attention between input data and learnable seed vectors
      to iteratively refine latent representations.

3. InducedPointFeaturizer:
    - Combines embedding, encoding, and decoding components to process input
      data and produce feature representations suitable for downstream tasks.
"""
import torch
from torch import nn

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block implementing multi-head attention,
    followed by a feed-forward network.
    """
    def __init__(self, d_model, num_heads, ffn_features=None):
        super().__init__()

        # Multi-head attention mechanism
        self.mha = nn.MultiheadAttention(embed_dim=d_model,
                                         num_heads=num_heads,
                                         batch_first=True)

        # Layer normalization for stabilizing training
        self.layer_norm = nn.LayerNorm(d_model)

        # Define feed-forward network (FFN) with optional feature size
        if ffn_features is None:
            ffn_features = d_model * 4  # Default expansion factor is 4

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_features),  # Expand features
            nn.GELU(),                         # Apply GELU activation
            nn.Linear(ffn_features, d_model)   # Reduce back to original feature size
        )

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Forward pass of the cross-attention block.
        Arguments:
            query, key, value: Inputs to the attention mechanism.
            key_padding_mask: Optional mask to ignore certain positions in the key.
        Returns:
            Output after applying multi-head attention and feed-forward network.
        """
        # Apply multi-head attention
        att_out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        # Add residual connection and apply layer normalization
        out = self.layer_norm(query + att_out)

        # Apply feed-forward network
        ffn_out = self.ffn(out)
        # Add residual connection and apply layer normalization again
        return self.layer_norm(out + ffn_out)


class Encoder(nn.Module):
    """
    Encoder module for iterative cross-attention
    between input data and learnable seed vectors.
    """
    def __init__(self,
                 num_seeds,
                 num_rounds,
                 d_model,
                 num_heads,
                 ffn_features=None):

        super().__init__()

        # Initialize learnable seed vectors
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, d_model))

        # Create a stack of cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(2 * num_rounds - 1):
            # Alternate layers for data and seed updates
            layer = CrossAttentionBlock(d_model=d_model,
                                        num_heads=num_heads,
                                        ffn_features=ffn_features)
            self.layers.append(layer)

    def forward(self, data, key_padding_mask):
        """
        Forward pass of the encoder.
        Arguments:
            data: Input data tensor.
            key_padding_mask: Mask indicating which elements
                in the data should be ignored.
        Returns:
            Latent representation encoded by the cross-attention mechanism.
        """
        # Tile the seed vectors for each sample in the batch
        code = self.seeds.expand(data.size(0), -1, -1).to(data.device)

        # Alternate attention between seeds and data
        for layer_id, layer in enumerate(self.layers):
            if layer_id % 2 == 0:
                # Attention: seeds attending to data
                code = layer(code, data, data, key_padding_mask=key_padding_mask)
            else:
                # Attention: data attending to seeds
                data = layer(data, code, code, key_padding_mask=None)

        # Return the encoded seed representations
        return code


class InducedPointFeaturizer(nn.Module):
    """
    Main module for inducing point-based feature
    extraction with cross-attention.
    """
    def __init__(self,
                 input_features,
                 query_features,
                 num_seeds,
                 num_rounds,
                 d_model,
                 num_heads):

        super().__init__()

        # Embedding layer to project input features to d_model dimension
        self.input_embedding = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Embedding layer to project query features to d_model dimension
        self.query_embedding = nn.Sequential(
            nn.Linear(query_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Encoder for cross-attention between input and seeds
        self.encoder = Encoder(num_seeds  = num_seeds,
                               num_rounds = num_rounds,
                               d_model    = d_model,
                               num_heads  = num_heads)

        # Decoder using multi-head attention for final processing
        self.decoder = nn.MultiheadAttention(embed_dim   = d_model,
                                             num_heads   = num_heads,
                                             batch_first = True)

        self.output_layer = nn.Sequential(nn.ReLU(),
                                          nn.Linear(d_model, query_features))

    def forward(self, data, query, key_padding_mask):
        """
        Forward pass of the InducedPointFeaturizer.
        Arguments:
            data: Input data tensor.
            query: Query tensor for decoding.
            key_padding_mask: Mask for input data to ignore certain positions.
        Returns:
            Output features processed by the decoder.
        """
        # Embed input data
        data = self.input_embedding(data)

        # Encode latent representation using cross-attention with seeds
        latent = self.encoder(data, key_padding_mask)

        # Embed query for decoding
        query = self.query_embedding(query)

        # Apply decoder attention between query and latent representation
        output, _ = self.decoder(query, latent, latent)

        return self.output_layer(output)
