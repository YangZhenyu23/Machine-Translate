import math
import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from dataset import PAD_IDX


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                       # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    基于 PyTorch nn.Transformer 的 Encoder-Decoder 翻译模型。
    所有权重均随机初始化，不加载任何预训练参数。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        # ---------- Embedding + Positional Encoding ----------
        self.src_embedding = nn.Embedding(src_vocab_size, d_model,
                                          padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model,
                                          padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # ---------- Transformer Core ----------
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # ---------- Output Projection ----------
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    # --------------------------------------------------------
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # --------------------------------------------------------
    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        """生成上三角因果 mask，阻止 decoder 看到未来位置。"""
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device), diagonal=1
        )

    @staticmethod
    def make_pad_mask(tokens, pad_idx=PAD_IDX):
        """(batch, seq_len) -> bool mask, True 表示 padding 位置。"""
        return tokens == pad_idx

    # --------------------------------------------------------
    def encode(self, src, src_key_padding_mask):
        """只跑 encoder，返回 memory，用于推理时复用。"""
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

    def decode(self, tgt, memory, tgt_mask, tgt_key_padding_mask,
               memory_key_padding_mask):
        """只跑 decoder，用于推理时逐步生成。"""
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    # --------------------------------------------------------
    def forward(self, src, tgt):
        """
        Parameters
        ----------
        src : (batch, src_len)  英文 token ids
        tgt : (batch, tgt_len)  中文 token ids

        Returns
        -------
        logits : (batch, tgt_len, tgt_vocab_size)
        """
        device = src.device

        # masks
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        src_pad_mask = self.make_pad_mask(src)
        tgt_pad_mask = self.make_pad_mask(tgt)

        # embeddings + positional encoding
        src_emb = self.pos_encoder(
            self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # transformer forward
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

        logits = self.fc_out(out)               # (batch, tgt_len, tgt_vocab)
        return logits


def build_model(src_vocab_size, tgt_vocab_size, device="cpu", **kwargs):
    """便捷工厂函数：构建模型并移动到指定设备。"""
    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        **kwargs,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")
    return model
