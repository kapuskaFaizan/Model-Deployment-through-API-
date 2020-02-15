import numpy as np
import pickle
import pandas as pd
import sqlite3
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv('/home/kapuska/Documents/loan_app/data/Loan.csv')
data.dropna(inplace = True)
feacol =['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','CoapplicantIncome','Loan_Amount_Term', 'Property_Area']
col = [i for i in data.columns if i not in ['Loan_ID','Loan_status','Credit_History'] ]

v = OneHotEncoder()
v.fit(data[feacol])

x = v.transform(data[feacol]).toarray()

lab = [i for i in data.Loan_status]
lab = [0 if x=='N' else x for x in lab]
y = [1 if x == 'Y' else x for x in lab]

Train_X, Test_X, Train_Y, Test_Y = train_test_split(x,y , test_size = 0.2)

md = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
md.fit(Train_X , Train_Y)
pred = md.predict(Test_X)
print(x)
print("SVM Accuracy Score -> ",accuracy_score(pred, Test_Y)*100)
print(classification_report(Test_Y,pred))
print(type(data['CoapplicantIncome'][0]))
new = [['Male','Yes', '0' , 'Graduate','No', 0.0 , 360.0 ,'Urban']]
enc_new = v.transform(new).toarray()
ans = md.predict(enc_new)
print(ans)
#pickle.dump(v, open("vectorizer.pickle", "wb"))
#pickle.dump(md, open("model.sav", "wb"))
#md.predict([])