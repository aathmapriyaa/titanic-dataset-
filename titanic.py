import pandas as pd 
import sklearn.linear_model as lm 
import sklearn.neighbors as knn
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
mydata = pd.read_csv("titanic.csv")
le = LabelEncoder()
mydata["Cabin_encoded"] = le.fit_transform(mydata[["Cabin"]]) 
mydata["Sex_encoded"] = le.fit_transform(mydata[["Sex"]]) 
x = mydata[["Pclass","Sex_encoded","Cabin_encoded"]]
y = mydata[["Survived"]]
print(mydata)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = knn.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy score = ", round(accuracy_score(y_test, y_pred) * 100, 2)) 
print(model.predict([[3,1,147]]))  