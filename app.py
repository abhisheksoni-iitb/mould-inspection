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

def analomalies(features):
    # if features[0] 

    if features[0] < 2.18 or features[0] > 4.03:
        return 1
    
    if features[1] < 1.5 or features[1] >2.63:
        return 1
    
    if features[2] < 32.4 or features[2] > 47.4:
        return 1
    
    if features[3] > 1600 or features[3] <1200:
        return 1
    
    if features[4] > 47 or features[4] < 42:
        return 1
    
    if features[5] > 59.5 or features[5] <0:
        return 1
    
    if features[6] > 22.94 or features[6] <0:
        return 1
    
    # if features[1] > 0 or features[1] <2.63:
    #     return 1
    
    if features[8] > 1650 or features[8] <1500:
        return 1

    if features[9] > 135 or features[9] <100:
        return 1

    if features[10] > 10.96 or features[10] <9.42:
        return 1

    if features[11] > 8.37 or features[11] <7.35:
        return 1

    if features[12] > 2.78 or features[12] <1.91:
        return 1

    if features[13] > 3.8 or features[13] <2.8:
        return 1

    if features[14] > 5.5 or features[14] <4.65:
        return 1

    return 0
    

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
    # print(features)

    condition = analomalies(features)

    if condition != 1:
        final_features = pd.DataFrame(data=dictionary)
        
        final_features = xgb.DMatrix(final_features.values)

        prediction = model.predict(final_features)

        output = round(prediction[0], 2)
    # output = len(features)

    else:
        output = 1

    if output < 0.5:
        output = 'Non-defective mould'
    else:
        output = 'Defective mould'


    return render_template('index.html', prediction_text='Prediction:  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)