import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
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
# You can use 'all-MiniLM-L6-v2' if your data is in English.
# For multilingual datasets, consider models like 'distiluse-base-multilingual-cased-v2'.
model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
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
# Create a pipeline that first scales the embeddings and then applies the SVM.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

# Set up hyperparameter grid to search over different kernels, C, and gamma values.
param_grid = {
    'svm__kernel': ['linear', 'rbf'],
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto']  # 'gamma' only relevant for 'rbf' kernel
}

# Perform grid search with 5-fold cross-validation.
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(train_embeddings, train_df['label'])

print("Best parameters found:", grid_search.best_params_)

# Use the best estimator from the grid search.
best_svm = grid_search.best_estimator_

# -------------------------
# 4. Evaluate the Best SVM Classifier
# -------------------------
# Evaluate on validation set
val_preds = best_svm.predict(val_embeddings)
val_acc = accuracy_score(val_df['label'], val_preds)
print("Validation Accuracy:", val_acc)
print("Validation Classification Report:")
print(classification_report(val_df['label'], val_preds, target_names=["negative", "neutral", "positive"]))

# Evaluate on test set
test_preds = best_svm.predict(test_embeddings)
test_acc = accuracy_score(test_df['label'], test_preds)
print("Test Accuracy:", test_acc)
print("Test Classification Report:")
print(classification_report(test_df['label'], test_preds, target_names=["negative", "neutral", "positive"]))
