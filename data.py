# data.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gensim
from utils import tokenize, encode_text

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, embedding_type, tokenizer=None, word_to_idx=None, max_len=50):
        self.texts = texts
        self.labels = labels
        self.embedding_type = embedding_type
        self.max_len = max_len
        if embedding_type == "fasttext":
            self.word_to_idx = word_to_idx
        else:
            self.tokenizer = tokenizer  # Hugging Face tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.embedding_type == "fasttext":
            encoded = torch.tensor(encode_text(text, self.word_to_idx, self.max_len), dtype=torch.long)
            return encoded, torch.tensor(label, dtype=torch.long)
        else:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            input_ids = encoding['input_ids'].squeeze()         # (max_len,)
            attention_mask = encoding['attention_mask'].squeeze() # (max_len,)
            return (input_ids, attention_mask), torch.tensor(label, dtype=torch.long)

def load_data(train_csv, val_csv, test_csv, label_mapping):
    """Load CSV files and map labels to integers."""
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    train_df['label'] = train_df['label'].map(label_mapping)
    val_df['label']   = val_df['label'].map(label_mapping)
    test_df['label']  = test_df['label'].map(label_mapping)

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def build_vocab(texts):
    """Build a vocabulary mapping from training texts."""
    vocab = {}
    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # Reserve 0 for padding
    return vocab

def create_embedding_matrix(word_to_idx, ft_model, embedding_dim):
    """Create an embedding matrix using pre-trained FastText vectors."""
    vocab_size = len(word_to_idx) + 1  # +1 for padding token
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_to_idx.items():
        if word in ft_model:
            embedding_matrix[idx] = ft_model[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix

def create_dataloaders(train_data, val_data, test_data, embedding_type, tokenizer, word_to_idx, max_len, batch_size):
    """Create PyTorch DataLoaders for the datasets."""
    train_dataset = SentimentDataset(train_data[0], train_data[1], embedding_type, tokenizer, word_to_idx, max_len)
    val_dataset   = SentimentDataset(val_data[0], val_data[1], embedding_type, tokenizer, word_to_idx, max_len)
    test_dataset  = SentimentDataset(test_data[0], test_data[1], embedding_type, tokenizer, word_to_idx, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def load_fasttext_model(ft_model_path, embedding_dim):
    """Load a pre-trained FastText model."""
    try:
        ft_model = gensim.models.KeyedVectors.load("wiki.id.kv", mmap='r')
        print("FastText vectors loaded quickly from binary file!")
    except Exception:
        ft_model = gensim.models.KeyedVectors.load_word2vec_format(ft_model_path, binary=False)
        ft_model.save("wiki.id.kv")
        print("FastText conversion complete: saved as 'wiki.id.kv'")
    return ft_model
