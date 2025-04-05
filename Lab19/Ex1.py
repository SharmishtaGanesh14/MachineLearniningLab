import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data=pd.read_csv("../datasets/heart.csv")
    # pd.set_option('display.max_columns',None)
    # print(data.isnull().sum())
    # print(data.head())
    X=data.drop(columns=["output"])
    y=data["output"]
    return X,y

def process_data(X,Xt,y,yt):
    scaling=StandardScaler()
    X_scaled=scaling.fit_transform(X)
    Xt_scaled=scaling.transform(Xt)
    return X,Xt,np.array(y),np.array(yt)

# confusion matrix for different thresholds
def evaluate_thresholds(threshold,y_probs,yt):
    y_pred=(y_probs>=threshold).astype(int)
    cm=confusion_matrix(yt,y_pred)
    tn,fp,fn,tp=cm.flatten()
    accuracy = accuracy_score(yt, y_pred)
    precision = precision_score(yt, y_pred)
    recall = recall_score(yt, y_pred)  # Sensitivity
    specificity = tn / (tn + fp)
    f1 = f1_score(yt, y_pred)
    print(f"Threshold: {threshold}")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Sensitivity (Recall): {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"F1 Score: {f1:.2f}\n")


def main():
    X,y=load_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=43)
    X_train, X_test, y_train, y_test=process_data(X_train,X_test,y_train,y_test)
    model=LogisticRegression(penalty='l2', solver="lbfgs", max_iter= 1000)
    model.fit(X_train,y_train)
    y_probs=model.predict_proba(X_test)[:,1]
    for thr in [0.3,0.5,0.7]:
        evaluate_thresholds(thr,y_probs,y_test)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main()