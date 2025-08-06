import re

class AceTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}

    def build_vocab(self, texts):
        tokens = set()
        for text in texts:
            tokens.update(self.tokenize(text))
        self.vocab = {token: idx for idx, token in enumerate(sorted(tokens))}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, 0) for token in tokens]

    def decode(self, ids):
        return " ".join([self.reverse_vocab.get(i, "<UNK>") for i in ids])
