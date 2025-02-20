# utils.py

import re
import random
import numpy as np
import torch

def tokenize(text):
    """Basic tokenization: lowercase and remove punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def encode_text(text, word_to_idx, max_len):
    """Encode a text string into a list of word indices."""
    tokens = tokenize(text)
    indices = [word_to_idx.get(word, 0) for word in tokens]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
