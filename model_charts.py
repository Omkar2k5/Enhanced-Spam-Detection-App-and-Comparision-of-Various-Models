import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def classification_report_as_dataframe(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df

# Function to plot accuracy comparison
def plot_accuracy_comparison(accuracies):
    plt.figure(figsize=(8, 5))
    models = list(accuracies.keys())
    scores = list(accuracies.values())
    sns.barplot(x=models, y=scores)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# Function to plot confusion matrix heatmap
def plot_confusion_matrix(confusion, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=["Predicted Ham", "Predicted Spam"], yticklabels=["Actual Ham", "Actual Spam"])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

# Function to plot classification report (precision, recall, F1-score)
def plot_classification_report(report_df, model_type):
    # Ensure the classification report is in the correct format
    report_df = report_df.transpose()

    # Plot the classification report as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df[['precision', 'recall', 'f1-score']], annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.title(f"Classification Report - {model_type}")
    plt.tight_layout()
    plt.show()
