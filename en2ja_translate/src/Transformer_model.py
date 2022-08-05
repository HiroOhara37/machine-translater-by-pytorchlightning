import math
from typing import Optional

import torch
import torch.nn as nn
from torch import float32, long
from torch.nn.init import xavier_uniform_
from torchtyping import TensorType

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_square_subsequent_mask(
    seq_len: int,
) -> TensorType["seq_len", "seq_len", float32]:
    mask: TensorType["seq_len", "seq_len", float32] = (
        torch.triu(torch.ones(seq_len, seq_len)) == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


def create_mask(
    src: TensorType["BATCH_SIZE", "length", long],
    tgt: TensorType["BATCH_SIZE", "length", long],
    pad_idx: int,
) -> tuple[
    TensorType["seq_len_src", "seq_len_src", bool],
    TensorType["seq_len_tgt", "seq_len_tgt", float32],
    TensorType["BATCH_SIZE", "length", bool],
    TensorType["BATCH_SIZE", "length", bool],
]:
    seq_len_src: int = src.shape[1]
    seq_len_tgt: int = tgt.shape[1]

    mask_src: TensorType[seq_len_src, seq_len_src, bool] = torch.zeros(
        (seq_len_src, seq_len_src)
    ).type(torch.bool)
    mask_tgt: TensorType[
        seq_len_tgt, seq_len_tgt, float
    ] = generate_square_subsequent_mask(seq_len_tgt)

    padding_mask_src: TensorType["BATCH_SIZE", "length", bool] = src == pad_idx
    padding_mask_tgt: TensorType["BATCH_SIZE", "length", bool] = tgt == pad_idx

    return (
        mask_src.to(DEVICE),
        mask_tgt.to(DEVICE),
        padding_mask_src.to(DEVICE),
        padding_mask_tgt.to(DEVICE),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position: TensorType[max_len, 1, float32] = torch.arange(
            0, max_len, dtype=float32
        ).unsqueeze(1)
        div_term: TensorType[d_model / 2, float32] = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pos_enc: TensorType[max_len, d_model, float32] = torch.zeros(max_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        positional_encoding: TensorType[1, max_len, d_model, float32] = pos_enc.unsqueeze(
            0
        )
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(
        self, token_emb: TensorType["batch_size", "length", "d_model", float32]
    ) -> TensorType["batch_size", "length", "d_model", float32]:
        seq_len: int = token_emb.size(1)
        token_emb = token_emb * math.sqrt(self.d_model)
        token_emb = token_emb + self.positional_encoding[:, :seq_len]

        return self.dropout(token_emb)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_layers: int = 3,
        max_len: int = 500,
    ) -> None:
        super(Transformer, self).__init__()
        # nn.Embedding: 語彙サイズ * 埋め込み次元
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, norm=nn.LayerNorm(d_model)
        )

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers, norm=nn.LayerNorm(d_model)
        )

        self.out = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self,
        src: TensorType["batch_size", "length", long],
        tgt: TensorType["batch_size", "length", long],
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        # memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        # memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> TensorType["batch_size", "length", "tgt_vocab_size", float32]:

        src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(
            src, tgt, pad_idx=0
        )

        src_emb: TensorType[
            "batch_size", "src_length", "d_model", float32
        ] = self.pos_encoder(self.src_embedding(src))

        tgt_emb: TensorType[
            "batch_size", "tgt_length", "d_model", float32
        ] = self.pos_encoder(self.tgt_embedding(tgt))

        memory = self.encoder(
            src=src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        model_output: TensorType[
            "batch_size", "tgt_length", "d_model", float32
        ] = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        output: TensorType[
            "batch_size", "tgt_length", "tgt_vocab_size", float32
        ] = self.out(model_output)

        return output

    def encode(self, src, src_mask):
        return self.encoder(self.pos_encoder(self.src_embedding(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.decoder(self.pos_encoder(self.tgt_embedding(tgt)), memory, tgt_mask)
