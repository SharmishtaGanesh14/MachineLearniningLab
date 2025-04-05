import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
def load_data():
    data=pd.read_csv("../Iris.csv")
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

X,y=load_data()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,stratify=y,random_state=423)
X_train,X_test,y_train,y_test,label=process_data(X_train,X_test,y_train,y_test)
kernels=['rbf', 'poly', 'linear']
for kernel in kernels:
    model=SVC(kernel=kernel)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"Classification report for Iris dataset (kernel method: {kernel}):")
    print(classification_report(y_test,y_pred,output_dict=False,target_names=label.classes_))
