
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split


"""# **Importing Dataset**"""

data = pd.read_csv("D:\Project 95\Money_Laundering_Dataset.csv")
data.shape


data=data.drop(["Unnamed: 0"],axis=1)

data["step"].fillna(data["step"].median(), inplace = True)

### nameOrig

data["nameOrig"].fillna(data["nameOrig"].mode(), inplace = True)

###NewbalanceOrig
value3=data['newbalanceOrig']+data["amount"]
data['oldbalanceOrg'].fillna(value3, inplace = True)

### OldbalanceOrg
value2 = data['oldbalanceOrg'] - data["amount"]
data['newbalanceOrig'].fillna(value2, inplace = True)

##nameDest
data['nameDest'].fillna(data['nameDest'].mode(),  inplace = True)

##oldbalanceDest
data['oldbalanceDest'].fillna(data['oldbalanceDest'].mean(),  inplace = True)


##isFraud
data['isFraud'].fillna(data['isFraud'].mode()[0],inplace=True)
##isFraud
data['isFlaggedFraud'].fillna(data['isFlaggedFraud'].mode()[0],inplace=True)

data.isna().sum()

"""Duplicate values"""

data.duplicated().sum()  ## no duplicate values



# High amount
## Here finding the supspicous Transaction which above thrdhold amount 
data['high'] = [1 if n>250000 else 0 for n in data['amount']]

data.head()

# Rapid Movement

## Transaction frequency for benficier account
''' 1 if the receiver receives money from many individuals else it will be 0'''
data['rapid']=data['nameDest'].map(data['nameDest'].value_counts())
data['Rapid']=[1 if n>30 else 0 for n in data['rapid']]
data.drop(['rapid'],axis=1,inplace = True)

''' customer ids which starts with C in Receiver name for cash_outs'''

def label_customer (row):
    if(row['nameDest'] and isinstance(row['nameDest'], str)):
        if row['type'] == 'CASH_OUT' and 'C' in row['nameDest']:
            return 1
    return 0
    
data['merchant'] = data.apply (lambda row: label_customer(row), axis=1)
data['merchant'].fillna(0,inplace=True)
data.head()


df = data.copy()

"""Balancing DataSet"""

df.head()

# One hot encoding
df =pd.concat([df,  pd.get_dummies(df['type'],    prefix='type_'  )],axis=1)
df.drop(['type'],  axis=1,  inplace = True)

df.head()

df.drop(['nameOrig', 'nameDest'], axis = 1, inplace = True)

#Normalization of  the numerical columns
col_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest','newbalanceDest']

def norm(i):
  x=(i-i.min())/(i.max()-i.min())
  return x

df[col_names]=norm(df[col_names])

df.head()

"""# **Model Building**

**Splitting Data**
"""

X = df.drop('isFraud', axis=1).values
y = df['isFraud'].values

X.shape

y.shape



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 111)


# Handling class imbalance using SMOTE based techniques

###pip install imblearn###
from imblearn.over_sampling import SMOTE

# oversampling the train dataset using SMOTE
smt = SMOTE()
#X_train, y_train = smt.fit_resample(X_train, y_train)
SX, Sy= smt.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(SX, Sy, train_size = 0.8, random_state = 111)



# Define Logistic Regression Model
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C= 20 ,penalty='l1',solver='liblinear',random_state=42,max_iter=1000)

# We fit our model with our train data
log.fit(X_train, y_train)


# Then predict results from X_test data
y_pred = log.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("accuracy_score : ", acc)
print("precision_score : ", prec)
print("recall_score : ", recall)
print("f1_score : ", f1)
print(cm)

### Pickle file

###### Creating Pickle File  ##########
import pickle
pickle.dump(log,open("project.pkl","wb"))



loaded_model=pickle.load(open("project.pkl","rb"))
output=loaded_model.predict(X_test)
