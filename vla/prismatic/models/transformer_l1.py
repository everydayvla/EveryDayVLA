import torch.nn as nn
import torch

class TransformerActionDecoder(nn.Module):
    def __init__(self, hidden_dim=256, n_heads=4, n_layers=3):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: (batch, tgt_seq_len, hidden_dim)
        # memory: (batch, src_seq_len, hidden_dim) ← from encoder
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)


if __name__ == '__main__':
    model = TransformerActionDecoder()
    tgt = torch.rand(20, 32, 256)
    memory = torch.rand(20, 32, 256)
    out = model(tgt, memory)
    print(f"out.shape: {out.shape}")
