
import pandas as pd
import pickle
import streamlit as st



# Set page config
st.set_page_config(page_title="AML",
                   page_icon= "💵",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items={"Get help":"mailto:giridharan9608@gmail.com",
                                "Report a Bug":"mailto:giridharan9608@gmail.com",
                                "About": "The AML project typically involves the collection and analysis of data to identify patterns and suspicious activities that may indicate illegal behavior. The project also includes the implementation of various regulatory compliance measures, such as customer due diligence (CDD), enhanced due diligence (EDD), and transaction monitoring.The AML project uses advanced technology such as machine learning to identify patterns and suspicious activities in large datasets. The project also requires the collaboration of various stakeholders, including financial institutions, law enforcement agencies, regulators, and other organizations."} )


st.title("Anti Money Laundering System")
loaded_model=pickle.load(open("project.pkl","rb"))

uploaded_file = st.file_uploader("Choose a file")
def main(): 
    
    if uploaded_file is not None:
        #read csv
        df1=pd.read_csv(uploaded_file,encoding='utf-8',on_bad_lines='skip')
        data=df1.copy()
        data=data.dropna()
        
        data=data.drop(["Unnamed: 0"],axis=1)
    
         # High amount
         ## Here finding the supspicous Transaction which above thrdhold amount 
        data['high'] = [1 if n>250000 else 0 for n in data['amount']]
    
    
         # Rapid Movement
    
        ## Transaction frequency for benficier account

        data['rapid']=data['nameDest'].map(data['nameDest'].value_counts())
        data['Rapid']=[1 if n>30 else 0 for n in data['rapid']]
        data.drop(['rapid'],axis=1,inplace = True)
    
    
        def label_customer (row):
            if(row['nameDest'] and isinstance(row['nameDest'], str)):
                if row['type'] == 'CASH_OUT' and 'C' in row['nameDest']:
                    return 1
                return 0
        
        data['merchant'] = data.apply (lambda row: label_customer(row), axis=1)
    
    
        # One hot encoding
        data =pd.concat([data,  pd.get_dummies(data['type'],    prefix='type_'  )],axis=1)
    

    
        data.drop(['nameOrig', 'nameDest','type'], axis = 1, inplace = True)
    
        #   Normalization of  the numerical columns
        col_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest','newbalanceDest']
    
        def norm(i):
            x=(i-i.min())/(i.max()-i.min())
            return x
    
        data[col_names]=norm(data[col_names])
        if st.button("Check"):
            prediction=[]
            for i in range(data.shape[0]):  
                
                value=loaded_model.predict(data.to_numpy())
                prediction.append(value[0])
            df=pd.DataFrame(prediction,columns=["Status"])
            result = pd.concat([df1, df], axis=1)
            result.to_csv('output.csv')
                  
       
    else:
         st.warning("you need to upload a csv")
     


    
    
    
    
if __name__=="__main__":  
    main()
    
