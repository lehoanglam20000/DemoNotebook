import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import datetime
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

data = []
for i in range(1,40000):
    data.append(np.random.permutation(33))
    
df = pd.DataFrame(np.array(data))
rows, cols = df.shape
def getfeature():
    names = []
    names.append('y')
    for i in range(1, cols):
        names.append("x" +str(i))
    return names  
df.columns = getfeature()

################################################

X_train, X_test, y_train,  y_test = train_test_split(df.drop('y', axis=1),df["y"], test_size=0.2)
print("Start Linear regressor... ")
model = LinearRegression()
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)
print('mean_squared_error: {:.4}'.format(mean_squared_error(y_test,y_predicted)))
pd.DataFrame([list(y_test),list(y_predicted)]).T.head(30)
