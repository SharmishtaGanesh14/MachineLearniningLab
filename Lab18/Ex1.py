import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
def plot_decision_boundary(model,title,X,y):
    X1_min,X1_max=min(X[:,0])-1,max(X[:,0])+1
    X2_min,X2_max=min(X[:,1])-1,max(X[:,1])+1
    xx=np.linspace(X1_min,X1_max,500)
    yy=np.linspace(X2_min,X2_max,500)
    xx,yy=np.meshgrid(xx,yy)
    z=np.c_[xx.ravel(),yy.ravel()]
    y_pred=model.predict(z).reshape(xx.shape)
    plt.contourf(xx,yy, y_pred, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
def main():
    data = {
        'x1': [6, 6, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14],
        'x2': [5, 9, 6, 8, 10, 2, 5, 10, 13, 5, 8, 6, 11, 4, 8],
        'label': ['Blue', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Red',
                  'Red', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']
    }
    df=pd.DataFrame(data)
    X=df.drop(columns=['label']).values
    y=df['label'].values
    label=LabelEncoder()
    y_encoded=label.fit_transform(y)
    svm_rbf=SVC(kernel='rbf',gamma='auto')
    svm_poly=SVC(kernel='poly',degree=3)
    svm_rbf.fit(X, y_encoded)
    svm_poly.fit(X, y_encoded)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(svm_rbf, "SVM with RBF Kernel",X,y)
    plt.subplot(1, 2, 2)
    plot_decision_boundary(svm_poly, "SVM with Polynomial Kernel (degree=3)",X,y)
    plt.tight_layout()
    plt.show()
if __name__=="__main__":
    main()