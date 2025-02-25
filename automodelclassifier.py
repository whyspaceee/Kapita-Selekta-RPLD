import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# -------------------------
# 1. Load the Data
# -------------------------
train_csv = "nusa-x-indo/train.csv"
val_csv   = "nusa-x-indo/valid.csv"
test_csv  = "nusa-x-indo/test.csv"

train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
test_df  = pd.read_csv(test_csv)

# Map string labels to integers
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
train_df['label'] = train_df['label'].map(label_mapping)
val_df['label']   = val_df['label'].map(label_mapping)
test_df['label']  = test_df['label'].map(label_mapping)

# -------------------------
# 2. Tokenize and Process Text
# -------------------------
model_name = "LazarusNLP/all-indo-e5-small-v4"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), label

train_dataset = TextDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = TextDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
test_dataset = TextDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------
# 3. Define Transformer Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# -------------------------
# 4. Train the Model
# -------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, attention_masks, labels in train_loader:
            inputs, attention_masks, labels = inputs.to(device), attention_masks.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=attention_masks).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

train_model(model, train_loader, val_loader, criterion, optimizer)

# -------------------------
# 5. Evaluate the Model
# -------------------------
def evaluate(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, attention_masks, labels in data_loader:
            inputs, attention_masks, labels = inputs.to(device), attention_masks.to(device), labels.to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_masks).logits
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

val_preds, val_labels = evaluate(model, val_loader)
test_preds, test_labels = evaluate(model, test_loader)

print("Validation Accuracy:", accuracy_score(val_labels, val_preds))
print("Test Accuracy:", accuracy_score(test_labels, test_preds))
