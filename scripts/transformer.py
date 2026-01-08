import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import PAD_IDX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        mask = input.ne(self.padding_idx).long()
        positions = torch.cumsum(mask, dim=0) * mask + self.padding_idx
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation="relu",
        normalize_before=True,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation_dropout = nn.Dropout(activation_dropout)

        self.activation = {"relu": F.relu, "gelu": F.gelu}[activation]

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        # Self attention block
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = residual + self.dropout(src)
        if not self.normalize_before:
            src = self.norm1(src)
        # Feed forward block
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.activation(self.linear1(src))
        src = self.activation_dropout(src)
        src = self.linear2(src)
        src = residual + self.dropout(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation="relu",
        normalize_before=True,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation_dropout = nn.Dropout(activation_dropout)

        self.activation = {"relu": F.relu, "gelu": F.gelu}[activation]

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        # self attention block
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = residual + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)
        # cross attention block
        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = residual + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)
        # feed forward block
        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.activation(self.linear1(tgt))
        tgt = self.activation_dropout(tgt)
        tgt = self.linear2(tgt)
        tgt = residual + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        src_vocab_size,
        trg_vocab_size,
        embed_dim,
        nb_heads,
        src_hid_size,
        src_nb_layers,
        trg_hid_size,
        trg_nb_layers,
        dropout_p,
        tie_trg_embed,
        src_c2i,
        trg_c2i,
        attr_c2i,
        label_smooth,
        **kwargs
    ):
        """
        init
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.nb_heads = nb_heads
        self.src_hid_size = src_hid_size
        self.src_nb_layers = src_nb_layers
        self.trg_hid_size = trg_hid_size
        self.trg_nb_layers = trg_nb_layers
        self.dropout_p = dropout_p
        self.tie_trg_embed = tie_trg_embed
        self.label_smooth = label_smooth
        self.src_c2i, self.trg_c2i, self.attr_c2i = src_c2i, trg_c2i, attr_c2i
        self.src_embed = Embedding(src_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.trg_embed = Embedding(trg_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.position_embed = SinusoidalPositionalEmbedding(embed_dim, PAD_IDX)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nb_heads,
            dim_feedforward=src_hid_size,
            dropout=dropout_p,
            attention_dropout=dropout_p,
            activation_dropout=dropout_p,
            normalize_before=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=src_nb_layers, norm=nn.LayerNorm(embed_dim)
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nb_heads,
            dim_feedforward=trg_hid_size,
            dropout=dropout_p,
            attention_dropout=dropout_p,
            activation_dropout=dropout_p,
            normalize_before=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=trg_nb_layers, norm=nn.LayerNorm(embed_dim)
        )
        self.final_out = Linear(embed_dim, trg_vocab_size)

    def generate_square_subsequent_mask(self, sz):
            dropout=self.dropout_p,
            attention_dropout=self.dropout_p,
            activation_dropout=self.dropout_p,
            normalize_before=True,
        )
        self.decoder = UniversalTransformerDecoder(
            decoder_layer,
            num_layers=self.trg_nb_layers,
            norm=nn.LayerNorm(self.embed_dim),
        )


class TagUniversalTransformer(TagTransformer, UniversalTransformer):
    pass


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m