import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load test data
test_df = pd.read_csv("nusa-x-indo/test.csv")  # Replace with actual test file path
test_labels = test_df.set_index("id")["label"]

test_path = "gemini"

print(f"using model {test_path}")
# Load test results
results_df = pd.read_csv(f"{test_path}.csv")  # Replace with actual results file path
predicted_labels = results_df.set_index("id")["sentiment"]

# Filter only matching IDs
test_labels = test_labels[test_labels.index.isin(predicted_labels.index)]
predicted_labels = predicted_labels[predicted_labels.index.isin(test_labels.index)]

# Generate classification report
report = classification_report(test_labels, predicted_labels, digits=2)

# Generate confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels, labels=["positive", "neutral", "negative"])

# Print evaluation results
print("Test Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)