# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 03:13:46 2020

@author: Ankit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


# preprocessing
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedShuffleSplit,KFold,StratifiedKFold



#models
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier


#Downloading dataset

data=pd.read_csv("faults.csv")
top_data=data.head(3)
data.info()
data.describe().T

#Count visualiztion of defects through plot to show the number of Defects belonging to each Defect Type
bumps=data[data.Bumps==1].count().get(key = 'Bumps')
other_faults=data[data.Other_Faults==1].count().get(key = 'Other_Faults')
stains=data[data.Stains==1].count().get(key = 'Stains')
dirtiness=data[data.Dirtiness==1].count().get(key = 'Dirtiness')
k_scatch=data[data.K_Scatch==1].count().get(key = 'K_Scatch')
z_scratch=data[data.Z_Scratch==1].count().get(key = 'Z_Scratch')
pastry=data[data.Pastry==1].count().get(key = 'Pastry')


objects = ('bumps', 'other_faults', 'dirtiness', 'stains', 'k_scatch', 'z_scratch', 'pastry')
y_pos = np.arange(len(objects))
defects = [bumps, other_faults, dirtiness, stains, k_scatch, z_scratch, pastry]
plt.bar(y_pos, defects, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Type of defects')
plt.title('Total count of defect as per defect type')
plt.xticks(rotation='vertical')
plt.show()

#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sn.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

#spliting data set into dependent and independent variables
x_data=data.iloc[:,:-7]
y_data=data.iloc[:,27:34]

#changing hot encoding to label encoding
y_label=np.array(y_data)
y_label=np.where(y_label==1)[1]
y_label=pd.DataFrame(y_label)

# Standardize the data-set
x_data_std=StandardScaler().fit_transform(x_data)
x_data_std = pd.DataFrame(x_data_std)


# # Principal Component Analysis
pca = PCA().fit(x_data_std)
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(1, 28, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
ax.grid(axis='x')
plt.show()

#TEST AND TRAIN VALUES FOR 95% VARIANCE
pca = PCA(n_components=0.95)
pca.fit(x_data_std)
x_data_std = pca.transform(x_data_std)
x_data_std = pd.DataFrame(x_data_std)
print("Number of features after PCA = ", x_data_std.shape[1])
corrmat_pca = pd.DataFrame(x_data_std).corr()
sn.heatmap(corrmat_pca, vmax=.8, square=True);
plt.show()






################
#LOGISTIC REGRESSION

#STRATIFIED
accuracy_strata_log=[]
logreg = LogisticRegression()
sss = StratifiedKFold(n_splits=10,random_state=None)
sss.get_n_splits(x_data_std,y_label)

for train_index, test_index in sss.split(x_data_std,y_label):
    x1_train,x1_test=x_data_std.iloc[train_index],x_data_std.iloc[test_index]
    y1_train,y1_test=y_label.iloc[train_index],y_label.iloc[test_index]
    logreg.fit(x1_train, y1_train)
    prediction=logreg.predict(x1_test)
    score=accuracy_score(prediction,y1_test)
    accuracy_strata_log.append(score)

print(accuracy_strata_log)
accuracy_strata_log=pd.DataFrame(accuracy_strata_log)
accuracy_strata_log_mean = np.mean(accuracy_strata_log)
print(accuracy_strata_log_mean)


#kfold

accuracy_kfold_log=[]
logreg = LogisticRegression()
sss = KFold(n_splits=10,random_state=None)
sss.get_n_splits(x_data_std,y_label)

for train_index, test_index in sss.split(x_data_std,y_label):
    x1_train,x1_test=x_data_std.iloc[train_index],x_data_std.iloc[test_index]
    y1_train,y1_test=y_label.iloc[train_index],y_label.iloc[test_index]
    logreg.fit(x1_train, y1_train)
    prediction=logreg.predict(x1_test)
    score=accuracy_score(prediction,y1_test)
    accuracy_kfold_log.append(score)

print(accuracy_kfold_log)
accuracy_kfold_log=pd.DataFrame(accuracy_kfold_log)
accuracy_kfold_log_mean = np.mean(accuracy_kfold_log)
print(accuracy_kfold_log_mean)

###############
#support vector machine

#stratifiedkfold

accuracy_strata_svm=[]
clf = SVC()
sss = StratifiedKFold(n_splits=10,random_state=None)
sss.get_n_splits(x_data_std,y_label)
for train_index, test_index in sss.split(x_data_std,y_label):
    x1_train,x1_test=x_data_std.iloc[train_index],x_data_std.iloc[test_index]
    y1_train,y1_test=y_label.iloc[train_index],y_label.iloc[test_index]
    clf.fit(x1_train, y1_train)
    prediction=clf.predict(x1_test)
    score=accuracy_score(prediction,y1_test)
    accuracy_strata_svm.append(score)

print(accuracy_strata_svm)
accuracy_strata_svm=pd.DataFrame(accuracy_strata_svm)
acc_svm_strata_mean = np.mean(accuracy_strata_svm)
print(acc_svm_strata_mean)

#kfold

accuracy_kfold_svm=[]
clf = SVC()
sss = KFold(n_splits=10,random_state=None)
sss.get_n_splits(x_data_std,y_label)
for train_index, test_index in sss.split(x_data_std,y_label):
    x1_train,x1_test=x_data_std.iloc[train_index],x_data_std.iloc[test_index]
    y1_train,y1_test=y_label.iloc[train_index],y_label.iloc[test_index]
    clf.fit(x1_train, y1_train)
    prediction=clf.predict(x1_test)
    score=accuracy_score(prediction,y1_test)
    accuracy_kfold_svm.append(score)

print(accuracy_kfold_svm)
accuracy_kfold=pd.DataFrame(accuracy_kfold_svm)
acc_svm_kfold_mean = np.mean(accuracy_kfold)
print(acc_svm_kfold_mean)



#############
#random forest

#k fold                
accuracy_kfold_random=[]
for m in range(150,200):
    random_forest=RandomForestClassifier(n_estimators= m,random_state=None)
    
    
    sss = KFold(n_splits=10,random_state=None)
    sss.get_n_splits(x_data_std,y_label)
    for train_index, test_index in sss.split(x_data_std,y_label):
        x1_train,x1_test=x_data_std.iloc[train_index],x_data_std.iloc[test_index]
        y1_train,y1_test=y_label.iloc[train_index],y_label.iloc[test_index]
        random_forest.fit(x1_train, y1_train)
        prediction=random_forest.predict(x1_test)
        score=accuracy_score(prediction,y1_test)
        accuracy_kfold_random.append(score)
        


mean_acc_table=[]
accuracy_kfold_random=pd.DataFrame(accuracy_kfold_random,dtype=float)
for i in range(0,500,10):
    find=accuracy_kfold_random.loc[i:i+9]
    mean=np.mean(find)
    mean_acc_table.append(mean)

mean_acc_table=np.array(mean_acc_table)        

plt.figure()        
l = range(150,200)
for j in range(len(l)):     
    plt.plot( l, mean_acc_table)
    plt.xlabel('Values of n_estimators')
    plt.ylabel('Accuracy score')
    plt.title('Variation of accuracy score with different n_estimators values in random forest while using kfol')
plt.show()

#statified k fold
accuracy_strata_random=[]
for m in range(150,200):
    random_forest=RandomForestClassifier(n_estimators= m,random_state=None)
    
    
    sss = StratifiedKFold(n_splits=10,random_state=None)
    sss.get_n_splits(x_data_std,y_label)
    for train_index, test_index in sss.split(x_data_std,y_label):
        x1_train,x1_test=x_data_std.iloc[train_index],x_data_std.iloc[test_index]
        y1_train,y1_test=y_label.iloc[train_index],y_label.iloc[test_index]
        random_forest.fit(x1_train, y1_train)
        prediction=random_forest.predict(x1_test)
        score=accuracy_score(prediction,y1_test)
        accuracy_strata_random.append(score)
        


mean_acc_table=[]
accuracy_strata_random=pd.DataFrame(accuracy_strata_random,dtype=float)
for i in range(0,500,10):
    find=accuracy_strata_random.loc[i:i+9]
    mean=np.mean(find)
    mean_acc_table.append(mean)

mean_acc_table=np.array(mean_acc_table)        

plt.figure()        
l = range(150,200)
for j in range(len(l)):     
    plt.plot( l, mean_acc_table)
    plt.xlabel('Values of n_estimators')
    plt.ylabel('Accuracy score')
    plt.title('Variation of accuracy score with different n_estimators values in random forest while using stratified-kfol')
plt.show()











