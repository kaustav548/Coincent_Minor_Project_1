import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("./data/iris.csv")
x = df.iloc[:,0:4].values
y = df.iloc[:,-1].values


#data preprocessing
#catagorie_features = ["species"]
#one_hot = OneHotEncoder()
#transformer = ColumnTransformer([("one_hot",
#                                  one_hot,
#                                  catagorie_features)],
#                                  remainder="passthrough")

transformer = LabelEncoder()
y=transformer.fit_transform(y)

#data split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2)

d = DecisionTreeClassifier()

d.fit(x_train,y_train)
y_pred = d.predict(x_test)

print(accuracy_score(y_pred,y_test)*100)