import pandas as pd
import numpy as np


# Steps I took to solve Kaggle- Titanic Problem

# IMPORT DATA
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

# DATA INFORMATION
s=shape(train_data)

train_data.head(10)
train_data.dtypes
train_data.info()
# column labels
colum_lab=train_data.columns
#summary of the data
train_data.describe()

# ---------------------------------------------------------------------------------------
# PROCESSING DATA 
# See that Cabin,Embarked and Age all have missing points with cabin having the most 
# missing points
#determine exact number of missing points for each feature
# Which other features have missing values 
d=[sum(pd.isnull(train_data[c])==1) for c in train_data]
print d


#delete passengerId,Ticket and also Cabin. Cabin has more missing points than given data therefore 
# would be difficult to fill. 
train_data=train_data.drop(['Ticket','PassengerId'], axis=1)
total_null=sum(pd.isnull(train_data['Cabin'])==1)
fract_ofTot=float(total_null)/s[0]
if fract_ofTot>0.5:
    del train_data['Cabin']  
 
# Convert string variable to numerical for Gender and Embarked
train_data.loc[train_data["Sex"] == "male", "Gender"] = 1
train_data.loc[train_data["Sex"] == "female", "Gender"] = 0
train_data.loc[train_data["Embarked"] == "S", "Embarked"] = 0
train_data.loc[train_data["Embarked"] == "C", "Embarked"] = 1
train_data.loc[train_data["Embarked"] == "Q", "Embarked"] = 2


#Fill in missing Embarked value
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
imp.fit(train_data.Embarked)
train_data.Embarked=imp.transform(train_data.Embarked).T.flatten()

 
# Function that extracts the titles of individuals from name and also creates a list to 
# store the number of unique titles
def extractTitles(data):
    title=[]
    for l in data:
        val=l.find(',')
        p=l.find('.')
        title.append(l[val+2:p])
        unq_title=unique(title)
    return title,unq_title
    
#call function and store titles in new column
train_data['NameTitle'], Unq_title = extractTitles(train_data.Name)


# Convert title to numerical data for use in training model
k=1
train_data['NumTitles']=train_data.NameTitle
for u in Unq_title:
    train_data['NumTitles']= where(train_data.NameTitle==u,k,train_data.NumTitles)
    k=k+1


# Missing age values
# Take 2 approaches to fill in missing data then determine which is the best and compare 
# to case of using non missing values only to train model -Cross validation
# Age has a total of 177 missing points 
missing_points=[ii for ii in range(len(train_data.Age)) if pd.isnull(train_data.Age[ii])]
notmiss_indx=[jj for jj in range(len(train_data.Age)) if jj not in missing_points]
# Approach 1. Random values to fill in missing age based on age range of the persons title
New_age=[]
title_ageRng=[]
for u in Unq_title: 
    if sum(train_data.NameTitle==u)> 1 and u!='Mlle': 
        title_ageRng.append(max(train_data.Age[train_data.NameTitle==u])-min(train_data.Age[train_data.NameTitle==u]))
    else: 
        title_ageRng.append(max(train_data.Age[train_data.NameTitle==u]))
    
for j in missing_points:
    New_age.append(random.rand()* title_ageRng[nonzero(Unq_title==train_data.NameTitle[j])[0][0]]+
    min(train_data.Age[train_data.NameTitle==train_data.NameTitle[j]]))

New_age=[int(r) for r in New_age]
    

# Approach 2. Use decision tree regression model to fill in missing values
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split

predictor_labels4Age=['Pclass','Fare','SibSp','Parch','Embarked','Gender','NumTitles']
X_4AgeAll=train_data[predictor_labels4Age]
X_4AgenonNull=X_4AgeAll.ix[notmiss_indx,:] 
y_4Age=train_data.Age[notmiss_indx]

X_tAge, X_testAge, y_tAge, y_testAge = train_test_split(X_4AgenonNull, y_4Age, 
test_size=0.4, random_state=0)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_1.fit(X_tAge, y_tAge)
print regr_1.score(X_testAge,y_testAge)

predicted_Age=regr_1.predict(X_4AgeAll.ix[missing_points,:])

#Create training set for each approach
train_data['RandAge']=train_data.Age
train_data.RandAge[missing_points]=New_age
train_data['ForRegAge']=train_data.Age
train_data.ForRegAge[missing_points]=predicted_Age

#Use random forest to compare different methods of filling missing data to see which is best
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
predictor_labels1=['Pclass','Gender','Fare','SibSp','Parch','Embarked','NumTitles','RandAge']
X_randAge=train_data[predictor_labels1]
predictor_labels2=['Pclass','Gender','Fare','SibSp','Parch','Embarked','NumTitles','ForRegAge']
X_ForAge=train_data[predictor_labels2]
y=train_data['Survived']

# Data with no missing points
X_full=train_data[['Pclass','Age','Gender','Fare','SibSp','Parch','Embarked','NumTitles']]
X_filt=X_full.ix[notmiss_indx,:]
y_filt = y[notmiss_indx]

#compare accuracy score of trained model for each method of filling in missing age
clf_Age = RandomForestClassifier(n_estimators=100)
scores_RandAge = cross_validation.cross_val_score(clf_Age, X_randAge, y, cv=5)
print scores_RandAge.mean()
scores_ForRegAge = cross_validation.cross_val_score(clf_Age, X_ForAge, y, cv=5)
print scores_ForRegAge.mean()
scores_Org = cross_validation.cross_val_score(clf_Age, X_filt, y_filt, cv=5)
print scores_Org.mean() # expect lowest score considering it has less training examples

## Filling in with random values based on person title performs similar to using decision 
# tree regression. Therefore just use random filling, hence use RandAge in place of Age


#----------------------------------------------------------------------------------------

# FEATURE SELECTION
#Determine the most important features 
#with random age
predLab=['Pclass','Gender','Fare','Parch','NumTitles','SibSp','Embarked','RandAge']
predR=train_data[predLab]
clfR = RandomForestClassifier(n_estimators=100)
clfR.fit(predR,resp)
feat_score=clfR.feature_importances_

#create dictionary and sort from the lowest to highest value
dict_feat=dict(zip(predLab, feat_score))
sorted(dict_feat.items(), key=lambda x: x[1])



# ----------------------------------------------------------------------------------------
# TRAIN MODEL - Random Forest

# Split data to train and test set
predictor_labels=['Gender','RandAge','Fare','NumTitles','Pclass','SibSp','Parch']
predictor_data=train_data[predictor_labels]
response_data=train_data['Survived']
pred_train, pred_test, resp_train, resp_test = cross_validation.train_test_split(predictor_data, 
response_data, test_size=0.4, random_state=0)



# Determine best parameters to use for training and then train
from sklearn import grid_search
parameters={'n_estimators':[10,50,100,150]}
rfc = RandomForestClassifier()
clf=grid_search.GridSearchCV(rfc,parameters)
clf = clf.fit(pred_train, resp_train)  


#validate (find number of incorrect our of total)
pred_survived=clf.predict(pred_test)
acc=sum(where(pred_survived!=resp_test,0,1))/float(len(resp_test))
### accuracy is 0.81


#-----------------------------------------------------------------------------------------
# PREDICTING
#prepare data for predicting

# Add features for name titles
test_data['NameTitle'], Unq_titleT = extractTitles(test_data.Name)

# Convert title to numerical data 
k=1
test_data['NumTitles']=test_data.NameTitle
for u in Unq_titleT:
    test_data['NumTitles']= where(test_data.NameTitle==u,k,test_data.NumTitles)
    k=k+1

#determine if there is any missing points and fill
s_test=shape(test_data)
d_test=[sum(pd.isnull(test_data[c])==1) for c in test_data]
##### age has 86 missing data points and fare 1

#fill missing fare value with random value between min and max
randFar=np.random.random(1)*(max(test_data.Fare)-min(test_data.Fare))
test_data.Fare[pd.isnull(test_data.Fare)==1]=randFar

#fill age appropriately using ranges for age from training data for respective titles
title_ageRngT=[]
for u in Unq_titleT: 
    if u in Unq_title:
        title_ageRngT.append(title_ageRng[np.where(Unq_title==u)[0][0]])
    else: 
        title_ageRngT.append(max(test_data.Age[test_data.NameTitle==u]))
        
#Create randomly generated missing values
missing_testIndx=[ii for ii in range(len(test_data.Age)) if pd.isnull(test_data.Age[ii])]
New_Tage=[]
for j in missing_testIndx:
    New_Tage.append(random.rand()* title_ageRng[np.where(Unq_titleT==test_data.NameTitle[j])[0][0]]+
    min(test_data.Age[np.where(test_data.NameTitle==test_data.NameTitle[j])[0][0]])) 

#Check if there is still any nan value
if any(isnan(New_Tage)):
    test_data.NameTitle[np.where(isnan(New_Tage))[0][0]]
#there is a missing value which corresponds to Mrs for this value, just use most frequent
#age
imp.fit(New_Tage)
New_Tage=imp.transform(New_Tage).flatten()


#Fill in missing values
test_data.Age[missing_testIndx]=New_Tage

# Convert string variable to numerical for Gender and Embarked
test_data.loc[test_data["Sex"] == "male", "Gender"] = 1
test_data.loc[test_data["Sex"] == "female", "Gender"] = 0
test_data.loc[test_data["Embarked"] == "S", "Embarked"] = 0
test_data.loc[test_data["Embarked"] == "C", "Embarked"] = 1
test_data.loc[test_data["Embarked"] == "Q", "Embarked"] = 2



#predict response
predictors4pred=test_data[['Gender','Age','Fare','NumTitles','Pclass','SibSp','Parch']]
test_data['SurvivedTrial']=clf.predict(predictors4pred)
test_data.to_csv('Results.csv',columns=['PassengerId','SurvivedTrial'],index=False)
  
  







    
