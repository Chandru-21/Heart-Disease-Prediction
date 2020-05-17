# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:56:09 2020

@author: Chandramouli
"""

#1-has heart disease
#0-doesn't
import pandas as pd

dataset= pd.read_csv('C://Users//Chandramouli//.spyder-py3//ML DATASETS//heart.csv')

total = dataset.isnull().sum().sort_values(ascending=False)#to find no of null values in each  column
#there are no null values
dataset.describe()
dataset.head()
x=dataset.iloc[:,:-1]#independant variables
x1=dataset.drop(['age','trestbps','chol','thalach'],1)
y=dataset.iloc[:,-1]#dependant variables
x.info()

#feature scaling
f=x.loc[:,['age','trestbps','chol','thalach']]
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
f1=scale.fit_transform(f)
f1=pd.DataFrame(f1)
f1.columns=['age','trestbps','chol','thalach']
x2=pd.concat([x1,f1],axis=1)


from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=2)

#logistic regression using cross validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
accuracy_logistic=[]
count=0
for train_index,test_index in skf.split(x2,y):
    count=count+1
    if(count<2):
        x_train,x_test =x2.iloc[train_index],x2.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=LogisticRegression()
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_logistic.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_logistic=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)*100

#normal method using logistic regression

from sklearn.model_selection import train_test_split

x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y,test_size=0.25,random_state=123)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train1,y_train1)

y_pred1=classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix
cm_logistic=confusion_matrix(y_test1,y_pred1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test1,y_pred1)*100

#k-nearest using cross validation

from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=4)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
accuracy_Knearest=[]
count=0
for train_index,test_index in skf.split(x2,y):
     count=count+1
     if(count<3):
    
        x_train,x_test =x2.iloc[train_index],x2.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_Knearest.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_Knearest=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100

#normal method using K nearest algorithm

from sklearn.model_selection import train_test_split

x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y,test_size=0.25,random_state=123)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train1,y_train1)

y_pred=classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix
cm_logistic=confusion_matrix(y_test1,y_pred1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test1,y_pred1)

##Naive Bayes using cross validation
#feature scaling is not needed
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=2)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
accuracy_NB=[]
count=0
for train_index,test_index in skf.split(x,y):
     count=count+1
     if(count<2):
        x_train,x_test =x.iloc[train_index],x.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=GaussianNB()
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_NB.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_NB=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100

#normal method using Naive baye's

from sklearn.model_selection import train_test_split

x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y,test_size=0.25,random_state=123)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train1,y_train1)
y_pred1=classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix
cm_NB=confusion_matrix(y_test1,y_pred1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test1,y_pred1)

###Decision tree
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=2)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
accuracy_DT=[]
count=0
for train_index,test_index in skf.split(x2,y):
    count=count+1
    if(count<2):
     
        x_train,x_test =x2.iloc[train_index],x2.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=DecisionTreeClassifier(criterion='entropy')
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_DT.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_DT=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100

#normal method using DT
from sklearn.model_selection import train_test_split

x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y,test_size=0.25,random_state=123)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train1,y_train1)

y_pred1=classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix
cm_DT=confusion_matrix(y_test1,y_pred1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test1,y_pred1)

#random forest
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=4)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
accuracy_RF=[]
count=0
for train_index,test_index in skf.split(x2,y):
    count=count+1
    if(count<2):
     
        x_train,x_test =x2.iloc[train_index],x2.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=RandomForestClassifier(n_estimators=10,criterion='entropy')
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_RF.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_RF=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100

#Random Forest with Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
parameter=[{'n_estimators':[10,20,30,40],'criterion':['gini','entropy'],'bootstrap':[True,False],'max_depth':[2,3,4,5,6]}]
grid_search=GridSearchCV(classifier,param_grid=parameter,scoring='accuracy',cv=4)
grid_search=grid_search.fit(x_train,y_train)
grid_search.best_params_
grid_search.best_score_
grid_search.cv_results_
grid_search.best_index_

#hyper parameter result 
classifier=RandomForestClassifier(n_estimators=30,criterion='gini',max_depth=3,bootstrap=True)

#using it in Random Forest algorithm

from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=4)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
accuracy_RF1=[]
count=0
for train_index,test_index in skf.split(x2,y):
    count=count+1
    if(count<2):
  
     
        x_train,x_test =x2.iloc[train_index],x2.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=RandomForestClassifier(n_estimators=30,criterion='gini',max_depth=3,bootstrap=True)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_RF1.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_RF1=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100


#inference
#logistic regression,decision tree,random forest all gave 100 percent accuracy using cross validation technique







    
        
    
        
