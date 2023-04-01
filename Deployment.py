
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def load_model(modelfile):
	loaded_model = pickle.load(open("xgb.pkl", 'rb'))
	return loaded_model

def norm(i):
  x=(i-i.min())/(i.max()-i.min())
  return x

#Surge indicator
def surge_indicator(data):
    '''Creates a new column which has 1 if the transaction amount is greater than the threshold
    else it will be 0'''
    data['isFlaggedFraud']=[1 if n>250000 else 0 for n in data['amount']]

st.title("Anti-Money Laundering Using Machine learning")
data = st.file_uploader("Upload an File", type=["csv"])

if st.button('Predict'):
    df1 = pd.read_csv(data)
    loaded_model = load_model('xgb.pkl')
    surge_indicator(df1)
    df2 = df1.copy()
    
    col_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest','newbalanceDest']
    df2[col_names]=norm(df2[col_names])

    df2=pd.concat([df2,pd.get_dummies(df2['type'], prefix='type_')],axis=1)
    df2.drop(['type'],axis=1,inplace = True)
    df2 = df2.drop(['nameOrig','nameDest'], axis=1)
    


    result = loaded_model.predict(df2)
    df1['isFlaggedFraud'] = df2['isFlaggedFraud']
    df1['isFraud'] = result

    st.dataframe(df1)