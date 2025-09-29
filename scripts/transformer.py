import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
PAD_IDX = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Missing classes that are referenced
class Embedding(nn.Embedding):
    pass

class Linear(nn.Linear):
    pass

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, pad_idx):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        
    def forward(self, x):
        # Simple positional encoding for now
        if x.dim() == 1:
            # Single sequence
            seq_len = x.size(0)
            pos = torch.arange(seq_len, device=x.device).float()
        else:
            # Batch of sequences
            seq_len = x.size(0)
            pos = torch.arange(seq_len, device=x.device).float()
        
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=x.device).float() * 
                           -(math.log(10000.0) / self.embed_dim))
        
        pe = torch.zeros(seq_len, self.embed_dim, device=x.device)
        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term)
        
        # Handle masking if needed
        if x.dim() > 1 and x.size(1) > 1:
            # Batch case - expand to match batch dimension
            pe = pe.unsqueeze(1).expand(-1, x.size(1), -1)
        
        return pe

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    pass

class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    pass

# PAD_IDX is defined above as 0

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
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=src_nb_layers, norm=nn.LayerNorm(embed_dim)
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nb_heads,
            dim_feedforward=trg_hid_size,
            dropout=dropout_p,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=trg_nb_layers, norm=nn.LayerNorm(embed_dim)
        )
        self.final_out = Linear(embed_dim, trg_vocab_size)
        if tie_trg_embed:
            self.final_out.weight = self.trg_embed.weight
        self.dropout = nn.Dropout(dropout_p)
        # self._reset_parameters()

    def embed(self, src_batch, src_mask):
        word_embed = self.embed_scale * self.src_embed(src_batch)
        pos_embed = self.position_embed(src_batch)
        embed = self.dropout(word_embed + pos_embed)
        return embed

    def encode(self, src_batch, src_mask):
        embed = self.embed(src_batch, src_mask)
        return self.encoder(embed, src_key_padding_mask=src_mask)

    def decode(self, enc_hs, src_mask, trg_batch, trg_mask):
        word_embed = self.embed_scale * self.trg_embed(trg_batch)
        pos_embed = self.position_embed(trg_batch)
        embed = self.dropout(word_embed + pos_embed)

        trg_seq_len = trg_batch.size(0)
        causal_mask = self.generate_square_subsequent_mask(trg_seq_len)
        dec_hs = self.decoder(
            embed,
            enc_hs,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=trg_mask,
            memory_key_padding_mask=src_mask,
        )
        return F.log_softmax(self.final_out(dec_hs), dim=-1)

    def forward(self, src_batch, src_mask, trg_batch, trg_mask):
        """
        only for training
        """
        src_mask = (src_mask == 0).transpose(0, 1)
        trg_mask = (trg_mask == 0).transpose(0, 1)
        # trg_seq_len, batch_size = trg_batch.size()
        enc_hs = self.encode(src_batch, src_mask)
        # output: [trg_seq_len, batch_size, vocab_siz]
        output = self.decode(enc_hs, src_mask, trg_batch, trg_mask)
        return output

    def count_nb_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def loss(self, predict, target, reduction=True):
        """
        compute loss
        """
        predict = predict.view(-1, self.trg_vocab_size)

        if not reduction:
            loss = F.nll_loss(
                predict, target.view(-1), ignore_index=PAD_IDX, reduction="none"
            )
            loss = loss.view(target.shape)
            loss = loss.sum(dim=0) / (target != PAD_IDX).sum(dim=0)
            return loss

        # nll_loss = F.nll_loss(predict, target.view(-1), ignore_index=PAD_IDX)
        target = target.reshape(-1, 1)
        non_pad_mask = target.ne(PAD_IDX)
        nll_loss = -predict.gather(dim=-1, index=target)[non_pad_mask].mean()
        smooth_loss = -predict.sum(dim=-1, keepdim=True)[non_pad_mask].mean()
        smooth_loss = smooth_loss / self.trg_vocab_size
        loss = (1.0 - self.label_smooth) * nll_loss + self.label_smooth * smooth_loss
        return loss

    def get_loss(self, data, reduction=True):
        src, src_mask, trg, trg_mask = data
        out = self.forward(src, src_mask, trg, trg_mask)
        loss = self.loss(out[:-1], trg[1:], reduction=reduction)
        return loss

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        # Use the device of the model's parameters instead of hardcoded DEVICE
        device = next(self.parameters()).device
        return mask.to(device)


class TagTransformer(Transformer):
    def __init__(self, *, nb_attr, **kwargs):
        super().__init__(**kwargs)
        self.nb_attr = nb_attr
        # 0 -> word character, 1 -> feature/tag
        self.special_embeddings = Embedding(2, self.embed_dim)

    def embed(self, src_batch, src_mask):
        # Get word embeddings for all tokens
        word_embed = self.embed_scale * self.src_embed(src_batch)
        
        # Create mask: 0 for word characters, 1 for features/tags
        # Features/tags are in the last nb_attr positions of vocabulary
        feature_mask = (src_batch >= (self.src_vocab_size - self.nb_attr)).long()
        
        # Get special embeddings based on the mask
        special_embed = self.embed_scale * self.special_embeddings(feature_mask)
        
        # Create positional embeddings where features get position 0
        # Only count positions for word characters
        seq_len = src_batch.size(0)
        batch_size = src_batch.size(1)
        
        # Initialize positional embeddings
        pos_embed = torch.zeros(seq_len, batch_size, self.embed_dim, device=src_batch.device)
        
        # Calculate character positions (excluding features)
        char_positions = torch.zeros(seq_len, batch_size, dtype=torch.long, device=src_batch.device)
        
        for b in range(batch_size):
            char_count = 1  # Start from position 1 for word characters
            for i in range(seq_len):
                if feature_mask[i, b] == 0:  # Word character
                    char_positions[i, b] = char_count
                    char_count += 1
                else:  # Feature/tag
                    char_positions[i, b] = 0  # Position 0 for features
        
        # Generate positional embeddings using the character positions
        pos_embed = self.position_embed(char_positions)
        
        # Combine all embeddings
        embed = self.dropout(word_embed + pos_embed + special_embed)
        return embed

