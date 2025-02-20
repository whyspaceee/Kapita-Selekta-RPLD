# models.py

import torch
import torch.nn as nn
from transformers import AutoModel

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Set to True if you want to fine-tune
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embeds = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        _, (hidden, _) = self.lstm(embeds)
        hidden = self.dropout(hidden[-1])
        out = self.fc(hidden)
        return out

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, hidden_dim, output_dim, freeze_bert=True):
        super(BERTClassifier, self).__init__()
        self.bert_model = bert_model
        print("freeze: ", freeze_bert)
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.bert_model.config.hidden_size,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        _, (hidden, _) = self.lstm(embeddings)
        hidden = self.dropout(hidden[-1])
        logits = self.fc(hidden)
        return logits

def get_model(embedding_type, config, ft_model=None, embedding_matrix=None, tokenizer=None):
    if embedding_type == "fasttext":
        vocab_size = embedding_matrix.shape[0]
        model = LSTMClassifier(vocab_size,
                               embedding_matrix.shape[1],
                               config.hidden_dim,
                               config.num_classes,
                               embedding_matrix)
    elif embedding_type == "indobert":
        indo_model = AutoModel.from_pretrained(config.indobert_model_name)
        model = BERTClassifier(indo_model,
                               config.hidden_dim,
                               config.num_classes,
                               freeze_bert=config.freeze_bert)
    elif embedding_type == "mbert":
        mbert_model = AutoModel.from_pretrained(config.mbert_model_name)
        model = BERTClassifier(mbert_model,
                               config.hidden_dim,
                               config.num_classes,
                               freeze_bert=config.freeze_bert)
    else:
        raise ValueError("Unsupported embedding type")
    return model
