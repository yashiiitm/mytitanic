from flask import Flask, request
import pandas as pd
import numpy as np
import json
import pickle
import os
app = Flask(__name__)
#load model amd scaler file
model_path = os.path.join(os.path.pardir,os.path.pardir,'models')
model_filepath = os.path.join(model_path,'lr_model_pkl')
scaler_filepath = os.path.join(model_path,'lr_scaler_pkl')
model = pickle.load(model_filepath)
scaler = pickle.load(scaler_filepath)
#columns
columns = [u'Age',u'Fare']
@app.route('/api',methods=['POST'])
def make_prediction():
    #read json object and convert to json string
    data = json.dumps(request.get_json(force=True))
    #create pandas dataframe using json string
    df = pd.read_json(data)
    #extract passenger IDs
    passenger_ids = df['PassengerId'].ravel()
    #actual survived values
    actuals = df['Survived'].ravel()
    #extract required columns and convert to matrix
    X = df[columns].as_matrix().astype('float')
    #transform the input
    X_scaled = scaler.transform(X)
    #make predictions
    predictions = model.predict(X_scaled)
    #create response dataframe
    df_response = pd.DataFrame({'PassengerId':passenger_ids,'Predicted':predictions,'Actual':actuals})
    #return json
    return df.response.to_json()

if __name__ = '__main__':
    app.run(port = 10001, debug = True)