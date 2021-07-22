import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb
import pandas as pd
import datetime
import joblib



app = Flask(__name__)
# model = pickle.load(open("pima.pickle.dat", "rb"))
# model = xgb.Booster()
# model.load_model("model.json")
# filename = 'global.model'
# model = joblib.load(open(filename, 'rb'))

model = xgb.Booster()
model.load_model('test_model.bin')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    
    print(len(features))
    
    dictionary = {
                    'Set_H2O_act':[features[0]],
                    'Meas_H2O':[features[1]],
                    'Meas_Temp':[features[2]],
                    'Weight':[features[3]],
                    'Set_Comp':[features[4]],
                    'Meas_Comp1':[features[5]],
                    'Lit_fin1':[features[6]],
                    'Lit_fin2':[features[7]],
                    'GCS':[features[8]],
                    'Permeability':[features[9]],
                    'total_Clay':[features[10]],
                    'active_clay':[features[11]],
                    'Dead_clay_':[features[12]],
                    'V_M_':[features[13]],
                    'LOI':[features[14]]
                }
    print(features)

    final_features = pd.DataFrame(data=dictionary)
    
    final_features = xgb.DMatrix(final_features.values)

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    # output = len(features)

    if output < 0.5:
        output = 'Non-defective mould'
    else:
        output = 'Defective mould'


    return render_template('index.html', prediction_text='Prediction:  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)