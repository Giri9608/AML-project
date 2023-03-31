import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_model(modelfile):
	loaded_model = pickle.load(open(xgb, 'rb'))
	return loaded_model

def std_scale(feature,mean,std):
     df2[feature] = (df2[feature]-mean)/std

#Surge indicator
def surge_indicator(data):
    '''Creates a new column which has 1 if the transaction amount is greater than the threshold
    else it will be 0'''
    data['isFlaggedFraud']=[1 if n>200000 else 0 for n in data['amount']]

st.title("Anti-Money Laundering Using Machine learning")
data = st.file_uploader("Upload an File", type=["csv", "excel"])

if st.button('Predict'):
    df1 = pd.read_csv(data)
    loaded_model = load_model('xgb.pkl')
    surge_indicator(df1)

    df2=pd.concat([df1,pd.get_dummies(df1['type'], prefix='type_')],axis=1)
    df2.drop(['type'],axis=1,inplace = True)
    df2 = df2.drop(['nameOrig','nameDest'], axis=1)
    
    std_scale('step',6.009223,3.609673)
    std_scale('amount',1.906158e+05,5.112169e+05)
    std_scale('oldbalanceOrg',5.682215e+05 ,2.049029e+06)
    std_scale('newbalanceOrig',5.142954e+05,2.035253e+06)
    std_scale('oldbalanceDest',6.999431e+05,2.050515e+06)
    std_scale('newbalanceDest',1.134943e+06,2.985611e+06)

    result = loaded_model.predict(df2)
    df1['isFlaggedFraud'] = df2['isFlaggedFraud']
    df2['isFraud'] = result

    st.dataframe(df1)
