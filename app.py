from flask import Flask, request, Response
import os
import pandas as pd
import json
import pickle

from empresa.empresa import PredictEmprestimo

model = pickle.load(open('model/final_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, this is the root endpoint. Your app is running!"


@app.route('/empresa/predict', methods=['POST'])
def emprestimo_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        pipeline = PredictEmprestimo()
        print('INICIO CLEANING')
        df_cleaning = pipeline.data_cleaning(test_raw)
        print('FIM CLEANING')
        print('INICIO FEATURE')
        df_feature = pipeline.feature_engineering(df_cleaning)
        print('FIM FEATURE')
        print('INICIO PREPARATION')
        df_preparation = pipeline.data_preparation(df_feature)
        print('FIM PREPARATION')
        print('INICIO PREDICT')

        pred = model.predict(df_preparation)
        #df_predict = pipeline.get_predictions(model, df_preparation)
        
        print('FIM PREDICT')
        return 'aaa'
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port=port)
