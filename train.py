import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

def train_epoch(model, loader, criterion, optimizer, device, embedding_type):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        inputs, targets = batch
        targets = targets.to(device)
        optimizer.zero_grad()
        if embedding_type in ["indobert", "mbert"]:
            input_ids, attention_mask = inputs
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
        else:
            inputs = inputs.to(device)
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    return epoch_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device, embedding_type, target_names=None):
    model.eval()
    epoch_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            targets = targets.to(device)

            if embedding_type in ["indobert", "mbert"]:
                input_ids, attention_mask = inputs
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, attention_mask)
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    class_report = classification_report(all_targets, all_predictions, target_names=target_names, zero_division=0)

    return epoch_loss / len(loader), accuracy, precision, recall, f1, conf_matrix, class_report

def train_and_evaluate(model, train_loader, val_loader, test_loader, config, embedding_type, device, writer: SummaryWriter, target_names=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    patience = config.early_stopping_patience
    counter = 0

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, embedding_type)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device, embedding_type, target_names)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:  {val_loss:.4f} | Val Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1-score: {val_f1:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Precision/Validation', val_prec, epoch)
        writer.add_scalar('Recall/Validation', val_rec, epoch)
        writer.add_scalar('F1-Score/Validation', val_f1, epoch)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered. No improvement in validation loss for {patience} consecutive epochs.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best model from epoch {best_epoch} loaded.")

    # Final evaluation on test set
    test_loss, test_acc, test_prec, test_rec, test_f1, test_conf_matrix, test_class_report = evaluate(model, test_loader, criterion, device, embedding_type, target_names)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1-score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_conf_matrix)
    print("Test Classification Report:")
    print(test_class_report)

    # Log test metrics
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', test_acc, epoch)
    writer.add_scalar('Precision/Test', test_prec, epoch)
    writer.add_scalar('Recall/Test', test_rec, epoch)
    writer.add_scalar('F1-Score/Test', test_f1, epoch)

    return model