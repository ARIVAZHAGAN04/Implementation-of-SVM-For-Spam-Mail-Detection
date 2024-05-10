# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Read the data frame using pandas.
3.Get the information regarding the null values present in the dataframe.
4.Split the data into training and testing sets.
5.Convert the text data into a numerical representation using CountVectorizer.
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7.Finally, evaluate the accuracy of the model.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ARIVAZHAGAN G R
RegisterNumber:  212223040020
*/
```

import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
*/


## Output:
![image](https://github.com/ARIVAZHAGAN04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161414455/34d13e1f-9ac9-4e8f-8421-c9dcb88780eb)
![image](https://github.com/ARIVAZHAGAN04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161414455/87d07e27-8ef7-4583-8727-b5930529b7dd)
![image](https://github.com/ARIVAZHAGAN04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161414455/b0413c26-cafb-4cd1-9fd9-6c42394e91cf)
![image](https://github.com/ARIVAZHAGAN04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161414455/22aab6da-6897-4dab-a5b9-2907a1b796d0)
![image](https://github.com/ARIVAZHAGAN04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161414455/428e0512-6d00-41e7-a4cc-ce1b6f40bed1)
![image](https://github.com/ARIVAZHAGAN04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161414455/07109087-c591-4929-aa06-46434f9bbe5a)
![image](https://github.com/ARIVAZHAGAN04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161414455/937e2032-adf3-44a3-affc-b8b5869a0cad)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
