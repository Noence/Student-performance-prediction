from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import matplotlib.pyplot as plt

  
# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 
  

## Useful info
x_cols = ['traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 
          'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 
          'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
midterm_grades = y[['G1', 'G2']]

y = y.drop(columns=['G1', 'G2'])
X_scoped = X[x_cols]

X_scoped=X_scoped.replace({'no': 0, 'yes': 1})
X_scoped = X_scoped.join(midterm_grades)
x_cols = X_scoped.columns
X_train, X_test, y_train, y_test = train_test_split(X_scoped, y, test_size=0.15, shuffle= True)


lineReg = LinearRegression()
lineReg.fit(X_train, y_train)
print('Score: ', lineReg.score(X_test, y_test))
weight_named = {x_col:val for (x_col, val) in zip(x_cols, np.abs(lineReg.coef_[0]))}
print(weight_named)

    

weights_sorted = sorted(weight_named, key=weight_named.get, reverse=True)[:1]
print(weights_sorted)
X_scoped = X_scoped[weights_sorted]
X_train, X_test, y_train, y_test = train_test_split(X_scoped, y, test_size=0.15, shuffle= True)
lineReg.fit(X_train, y_train)
print('Score: ', lineReg.score(X_test, y_test))

predictions = lineReg.predict(X_test)
plt.scatter(predictions, y_test['G3'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()
