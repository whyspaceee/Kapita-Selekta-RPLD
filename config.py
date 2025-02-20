# config.py

class Config:
    def __init__(self):
        # Embedding options: "fasttext", "indobert", or "mbert"
        self.embedding_type = "indobert"
        
        # Paths for FastText model and CSV data
        self.fasttext_path = "wiki.id.vec"  # Path to your FastText .vec file
        self.train_csv = "nusa-x-indo/train.csv"
        self.val_csv   = "nusa-x-indo/valid.csv"
        self.test_csv  = "nusa-x-indo/test.csv"
        
        # For BERT-based models
        self.indobert_model_name = "indolem/indobert-base-uncased"
        self.mbert_model_name = "bert-base-multilingual-cased"
        
        # Data processing and model parameters
        self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        self.max_len = 50           # Maximum sequence length
        self.hidden_dim = 128
        self.num_classes = 3
        self.batch_size = 32
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.num_epochs = 100
        self.random_seed = 42
        
        # If True, freeze BERT parameters
        self.freeze_bert = True

        # Early stopping: Stop if no improvement in validation loss for these many epochs
        self.early_stopping_patience = 10
