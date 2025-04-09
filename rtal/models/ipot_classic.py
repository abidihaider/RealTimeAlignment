"""
Induced-Point Operator Transformer model
with Fourier Positional Embedding
"""

import torch
from torch import nn


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

class PositionEmbedding(nn.Module):
    """
    Similar to transformer's position encoding,
    but generalizes it to arbitrary dimensions.

    Args:
        n_dims: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self,
                 n_dims,
                 d_model,
                 temperature = 10000,
                 scale       = None):

        super().__init__()

        self.num_pos_feats = (d_model // n_dims // 2) * 2
        self.padding = d_model - self.num_pos_feats * n_dims

        self.n_dims = n_dims
        self.temperature = temperature

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * torch.pi

    def forward(self, coords):
        """
        Args:
            coords: Point positions (*, in_dimensions)

        Returns:
            pos_emb (*, out_dimensions)
        """

        assert coords.shape[-1] == self.n_dims,\
            f"input dimention = {coords.shape} n_dims = {self.n_dims}"

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=coords.device)

        exponent = 2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats
        dim_t = self.temperature ** exponent

        coords = coords * self.scale
        pos_divided = coords.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*coords.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = nn.functional.pad(pos_emb, (0, self.padding))
        return pos_emb


class Attention(nn.Module):
    """
    Multihead attention with input/output transform
    """
    def __init__(self,
                 query_channels,
                 num_heads,
                 embed_dim,
                 dropout,
                 context_channels=None,
                 output_channels=None):

        super().__init__()

        context_dim = query_channels
        if context_channels is not None:
            context_dim = context_channels

        output_dim = query_channels
        if output_channels is not None:
            output_dim = output_channels

        self.mha = nn.MultiheadAttention(embed_dim   = embed_dim,
                                         num_heads   = num_heads,
                                         dropout     = dropout,
                                         batch_first = True)

        self.to_q = nn.Linear(query_channels, embed_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, embed_dim * 2, bias=False)

        self.to_out = nn.Linear(embed_dim, output_dim)

    def forward(self, query, context=None, key_padding_mask=None):

        if context is None:
            context = query

        query = self.to_q(query)
        key, value = self.to_kv(context).chunk(2, dim=-1)

        attn_out, attn_weights = self.mha(query, key, value,
                                          key_padding_mask=key_padding_mask)

        return self.to_out(attn_out), attn_weights


class GatedLinear(nn.Module):
    """
    A network with two linear layers and a GELU gate
    """
    def __init__(self,
                 channels,
                 multiplier,
                 dropout,
                 out_channels=None):

        super().__init__()

        if out_channels is None:
            out_channels = channels

        self.lin_1 = nn.Linear(channels, channels * multiplier * 2)
        self.lin_2 = nn.Linear(channels * multiplier, out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        data = self.lin_1(data)
        chunk_1, chunk_2 = data.chunk(2, dim=-1)
        data = chunk_1 * self.gelu(chunk_2)
        data = self.dropout(data)
        return self.lin_2(data)


class Encoder(nn.Module):
    """
    The Encoder has a single cross-attention block.
    It trains a Transformer kernel that summarized
    the inputs.
    """
    def __init__(self, *,
                 input_channels,
                 latent_channels,
                 kernel_size,
                 kernel_scale,
                 num_heads,
                 dropout):

        super().__init__()

        self.kernel = nn.Parameter(torch.randn(1, kernel_size, latent_channels)
                                   * kernel_scale)

        self.attn = Attention(query_channels   = latent_channels,
                              context_channels = input_channels,
                              embed_dim       = latent_channels,
                              num_heads       = num_heads,
                              dropout         = dropout)

        self.query_norm = nn.LayerNorm(latent_channels)
        self.context_norm = nn.LayerNorm(input_channels)

    def forward(self, inputs, key_padding_mask):

        kernel = self.kernel.expand(inputs.size(0), -1, -1).to(inputs.device)

        attn_out, _ = self.attn(query            = self.query_norm(kernel),
                                context          = self.context_norm(inputs),
                                key_padding_mask = key_padding_mask)
        return attn_out + kernel


class Processor(nn.Module):
    """
    Processor is several blocks of self-attention
    and feed-forward network.
    """
    def __init__(self, *,
                 num_attn_blocks,
                 latent_channels,
                 num_heads,
                 attn_dropout,
                 ffn_dropout,
                 multiplier):

        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(num_attn_blocks):
            # attention
            attn_norm = nn.LayerNorm(latent_channels)
            attn = Attention(query_channels = latent_channels,
                             embed_dim     = latent_channels,
                             num_heads     = num_heads,
                             dropout       = attn_dropout)

            # feed-forward network
            ffn_norm = nn.LayerNorm(latent_channels)
            ffn = GatedLinear(latent_channels,
                              multiplier = multiplier,
                              dropout    = ffn_dropout)

            self.layers.append(nn.ModuleList([attn_norm, attn,
                                              ffn_norm,  ffn]))

    def forward(self, latent):
        for attn_norm, attn, ffn_norm, ffn in self.layers:
            attn_out, _ = attn(attn_norm(latent))
            latent = attn_out + latent
            latent = ffn(ffn_norm(latent)) + latent
        return latent


class Decoder(nn.Module):
    """
    Decoder has one cross-attention for featurization
    and one feed-forward network for (pre)training output.
    """
    def __init__(self, *,
                 query_channels,
                 latent_channels,
                 out_channels,
                 num_heads,
                 attn_dropout,
                 ffn_dropout,
                 multiplier,
                 output_scale):

        super().__init__()

        # attention block for featurization
        self.attn = Attention(query_channels   = query_channels,
                              context_channels = latent_channels,
                              output_channels  = latent_channels,
                              embed_dim       = latent_channels,
                              num_heads       = num_heads,
                              dropout         = attn_dropout)
        self.query_norm = nn.LayerNorm(query_channels)
        self.content_norm = nn.LayerNorm(latent_channels)

        # feed-forward block for (pre)training output
        self.ffn = GatedLinear(latent_channels,
                               multiplier  = multiplier,
                               dropout     = ffn_dropout,
                               out_channels = out_channels)

        self.ffn_norm = nn.LayerNorm(latent_channels)

        self.output_scale = output_scale

    def forward(self, query, latent):

        # attention output (the features of the queires)
        attn_out, attn_weights = self.attn(query   = self.query_norm(query),
                                           context = self.content_norm(latent))

        feature = attn_out.clone()

        # (pre)training output
        output = self.ffn(self.ffn_norm(attn_out))

        return output * self.output_scale, feature, attn_weights



class IPOT(nn.Module):
    """
    Induced Point Operator Transformer model.
    The model has four parts:
        - Positional embedding
        - Encoder: Train a dataset-wise Transformer kernel
            to summarize the input.
        - Processor: processe the summarized input (latent).
        - Decoder: process the decoder queries.
    """
    def __init__(self,
                 encoder,
                 processor,
                 decoder,
                 input_dims,
                 input_channels,
                 query_dims,
                 query_channels):

        super().__init__()

        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder

        self.input_pos_emb = PositionEmbedding(n_dims  = input_dims,
                                               d_model = input_channels)
        self.query_pos_emb = PositionEmbedding(n_dims  = query_dims,
                                               d_model = query_channels)

    def forward(self, inputs, key_padding_mask, query):
        # embed the inputs and queries
        encoder_input = self.input_pos_emb(inputs)
        decoder_query = self.query_pos_emb(query)

        # run the model
        latent = self.encoder(encoder_input,
                              key_padding_mask=key_padding_mask)

        latent = self.processor(latent)

        output, feature, attn_weights = self.decoder(decoder_query, latent)

        return {'output'  : output,
                'latent'  : latent,
                'feature' : feature,
                'attn_weights': attn_weights}
