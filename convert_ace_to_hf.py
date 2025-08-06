import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

# Your model class
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

# Load your model
model_path = "ace_assistant_model.pt"
model = AceAssistantModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Hugging Face GPT2 config
config = GPT2Config(
    vocab_size=30522,
    n_positions=128,
    n_ctx=128,
    n_embd=256,
    n_layer=2,
    n_head=4
)

# Convert to GPT2LMHeadModel
hf_model = GPT2LMHeadModel(config)
with torch.no_grad():
    hf_model.transformer.wte.weight.copy_(model.embedding.weight)
    hf_model.lm_head.weight.copy_(model.fc.weight)

# Save the HF format
hf_model.save_pretrained("ace_hf_model")
config.save_pretrained("ace_hf_model")

print("✅ ACE model saved in Hugging Face format at: ace_hf_model/")
