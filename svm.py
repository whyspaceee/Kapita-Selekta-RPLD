import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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
# 2. Generate Sentence Embeddings with SBERT
# -------------------------
model_name = "distiluse-base-multilingual-cased-v2"
sbert_model = SentenceTransformer(model_name)

print("Encoding training data...")
train_embeddings = sbert_model.encode(train_df['text'].tolist(), show_progress_bar=True)
print("Encoding validation data...")
val_embeddings = sbert_model.encode(val_df['text'].tolist(), show_progress_bar=True)
print("Encoding test data...")
test_embeddings = sbert_model.encode(test_df['text'].tolist(), show_progress_bar=True)

# -------------------------
# 3. Build an Improved SVM Classifier Using a Pipeline and Grid Search
# -------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

param_grid = {
    'svm__kernel': ['linear', 'rbf'],
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(train_embeddings, train_df['label'])

print("Best parameters found:", grid_search.best_params_)

best_svm = grid_search.best_estimator_

# -------------------------
from sklearn.metrics import confusion_matrix

# Function to print confusion matrix
def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("    " + "  ".join(labels))
    for label, row in zip(labels, cm):
        print(f"{label} {row}")

# Evaluate on validation set
val_preds = best_svm.predict(val_embeddings)
val_acc = accuracy_score(val_df['label'], val_preds)
print("Validation Accuracy:", val_acc)
print("Validation Classification Report:")
print(classification_report(val_df['label'], val_preds, target_names=["negative", "neutral", "positive"]))
print_confusion_matrix(val_df['label'], val_preds, ["negative", "neutral", "positive"])

# Evaluate on test set
test_preds = best_svm.predict(test_embeddings)
test_acc = accuracy_score(test_df['label'], test_preds)
print("\nTest Accuracy:", test_acc)
print("Test Classification Report:")
print(classification_report(test_df['label'], test_preds, target_names=["negative", "neutral", "positive"]))
print_confusion_matrix(test_df['label'], test_preds, ["negative", "neutral", "positive"])
