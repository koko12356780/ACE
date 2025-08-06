import torch
import torch.nn as nn

class AceAssistantModel(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=256, num_heads=4, num_layers=2, seq_len=128):
        super(AceAssistantModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.fc(x)
        return logits
