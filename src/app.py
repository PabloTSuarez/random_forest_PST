from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pickle import dump  

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')
df

X = df.drop('Outcome',axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train

model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred

accuracy_score(y_test,y_pred)

hyperparametros = {
    'criterion':['gini','entropy'],
    'max_depth':[None,5,10,20],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'n_estimators':[60,120,150,200]
}

grid = GridSearchCV(model,hyperparametros,scoring='accuracy',cv=5)
grid

grid.fit(X_train,y_train)
grid.best_params_

mejor_modelo = grid.best_estimator_

y_best_pred = mejor_modelo.predict(X_test)

accuracy_score(y_test,y_best_pred)

b_model = RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_leaf=4,
                       n_estimators=120, random_state=42,bootstrap=False)


b_model.fit(X_train,y_train)
y_pred_b = b_model.predict(X_test)
accuracy_score(y_test,y_pred_b)

dump(mejor_modelo, open("../models/mejor_modelo_random_forest.sav","wb"))