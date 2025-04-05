import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import numpy as np
def load_data():
    data=pd.read_csv("../datasets/Iris.csv")
    # print(data.isnull().sum())
    # print(data.head())
    data = data[data["Species"].isin(["Iris-setosa", "Iris-versicolor"])]
    X = data[["SepalLengthCm", "SepalWidthCm"]]
    y = data["Species"]
    return X,y

def process_data(X,Xt,y,yt):
    label=LabelEncoder()
    y_encoded_train=label.fit_transform(y)
    y_encoded_test=label.transform(yt)
    scaling=StandardScaler()
    X_scaled_train=scaling.fit_transform(X)
    X_scaled_test=scaling.transform(Xt)
    return X_scaled_train,X_scaled_test,y_encoded_train,y_encoded_test,label

def plotting(model,title,x,y):
    x1=x[:,0]
    x2=x[:,1]
    x1_min,x1_max=min(x1),max(x1)
    x2_min,x2_max=min(x2),max(x2)
    xx=np.linspace(x1_min,x1_max,500)
    yy=np.linspace(x2_min,x2_max,500)
    xx,yy=np.meshgrid(xx,yy)
    z=np.c_[xx.ravel(),yy.ravel()]
    y_pred_grid=model.predict(z).reshape(xx.shape)
    plt.contourf(xx,yy,y_pred_grid,alpha=0.3,cmap=plt.cm.coolwarm)
    plt.scatter(x1,x2, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")

X,y=load_data()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,stratify=y,random_state=423)
X_train,X_test,y_train,y_test,label=process_data(X_train,X_test,y_train,y_test)
kernels=['rbf','poly', 'linear']
plt.figure(figsize=(12, 5))
for idx,kernel in enumerate(kernels):
    model=SVC(kernel=kernel,gamma=0.5,C=2)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    plt.subplot(1, 3, idx+1)
    plotting(model,f"SVC- Kernel method: {kernel}",X_train,y_train)
    print(f"Classification report for Iris dataset (kernel method: {kernel}):")
    print(classification_report(y_test,y_pred,output_dict=False,target_names=label.classes_))
plt.show()