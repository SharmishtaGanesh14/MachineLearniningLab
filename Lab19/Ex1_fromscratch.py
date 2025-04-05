import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def confusion_elements(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def compute_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    return accuracy, precision, sensitivity, specificity, f1

def plot_roc_auc(y_true, y_probs):
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred_thresh = (y_probs >= threshold).astype(int)
        tp, tn, fp, fn = confusion_elements(y_true, y_pred_thresh)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list, color='darkorange', label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve from Scratch')
    plt.grid(True)
    plt.legend()
    plt.show()
    # Sort FPR and TPR in ascending order of FPR
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    sorted_indices = np.argsort(fpr_arr)
    auc_score = np.trapezoid(tpr_arr[sorted_indices], fpr_arr[sorted_indices])
    return auc_score

def evaluate_heart_classifier():
    # Load and split data
    df = pd.read_csv("../datasets/heart.csv")
    X = df.drop("output", axis=1)
    y = df["output"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Get predictions
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)

    # Evaluate
    tp, tn, fp, fn = confusion_elements(y_test, y_pred)
    accuracy, precision, sensitivity, specificity, f1 = compute_metrics(tp, tn, fp, fn)
    auc_score = plot_roc_auc(y_test, y_probs)

    # Display
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Sensitivity  : {sensitivity:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC (manual) : {auc_score:.4f}")

if __name__ == "__main__":
    evaluate_heart_classifier()
