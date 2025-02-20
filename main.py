# main.py

import argparse
import torch
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from config import Config
from utils import set_seed, encode_text
from data import load_data, build_vocab, create_embedding_matrix, create_dataloaders, load_fasttext_model
from models import get_model
from train import train_and_evaluate

def main(args):
    # Initialize configuration and set seed for reproducibility
    config = Config()
    config.embedding_type = args.embedding_type
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    set_seed(config.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Load dataset
    train_data, val_data, test_data = load_data(
        config.train_csv, config.val_csv, config.test_csv, config.label_mapping
    )

    tokenizer = None
    word_to_idx = None
    embedding_matrix = None

    if config.embedding_type == "fasttext":
        # Load FastText model and build vocabulary
        ft_model = load_fasttext_model(config.fasttext_path, embedding_dim=300)
        config.embedding_dim = ft_model.vector_size
        word_to_idx = build_vocab(train_data[0])
        embedding_matrix = create_embedding_matrix(word_to_idx, ft_model, config.embedding_dim)
    else:
        # For BERT-based models, load the corresponding tokenizer
        model_name = config.indobert_model_name if config.embedding_type == "indobert" else config.mbert_model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config.embedding_dim = 768  # Typically BERT hidden size

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        config.embedding_type,
        tokenizer,
        word_to_idx,
        config.max_len,
        config.batch_size
    )

    # Get model
    model = get_model(config.embedding_type, config,
                      ft_model if config.embedding_type=="fasttext" else None,
                      embedding_matrix,
                      tokenizer)
    model.to(device)

    # Train and evaluate
    model = train_and_evaluate(model, train_loader, val_loader, test_loader, config, config.embedding_type, device, writer)

    # Inference example
    model.eval()
    sample_text = "layanan ini sangat buruk dan mengecewakan"
    with torch.no_grad():
        if config.embedding_type in ["indobert", "mbert"]:
            encoding = tokenizer.encode_plus(
                sample_text,
                add_special_tokens=True,
                max_length=config.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            output = model(input_ids, attention_mask)
        else:
            encoded = torch.tensor([encode_text(sample_text, word_to_idx, config.max_len)], dtype=torch.long).to(device)
            output = model(encoded)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f"Predicted sentiment for '{sample_text}': {predicted_class}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis with Different Embedding Options and TensorBoard Logging")
    parser.add_argument("--embedding_type", "--embedding", type=str, default="fasttext", choices=["fasttext", "indobert", "mbert"],
                        help="Type of embedding to use: fasttext, indobert, or mbert")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log_dir", type=str, default="runs/experiment", help="Directory for TensorBoard logs")
    args = parser.parse_args()
    main(args)
