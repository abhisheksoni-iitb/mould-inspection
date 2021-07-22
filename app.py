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

# model = xgb.Booster()
# model.load_model('test_model.bin')

model = xgb.XGBClassifier()
model.load_model("model_sklearn.json")

def analomalies(features):
    # if features[0] 

    
    if features[0] < 1.5 or features[0] >2.63:
        return 1
    
    if features[1] < 32.4 or features[1] > 47.4:
        return 1
    
    if features[2] > 1600 or features[2] <1200:
        return 1
    
    if features[3] > 47 or features[3] < 42:
        return 1
    
    if features[4] > 59.5 or features[4] <0:
        return 1
    
    if features[5] > 22.94 or features[5] <0:
        return 1
    
    
    if features[6] > 1650 or features[6] <1500:
        return 1

    if features[7] > 135 or features[7] <100:
        return 1

    if features[8] > 10.96 or features[8] <9.42:
        return 1

    if features[9] > 8.37 or features[9] <7.35:
        return 1

    if features[10] > 2.78 or features[10] <1.91:
        return 1

    if features[11] > 3.8 or features[11] <2.8:
        return 1

    if features[12] > 5.5 or features[12] <4.65:
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
                    'Meas_H2O':[features[0]],
                    'Meas_Temp':[features[1]],
                    'Weight':[features[2]],
                    'Set_Comp':[features[3]],
                    'Meas_Comp1':[features[4]],
                    'Lit_fin1':[features[5]],
                    'GCS':[features[6]],
                    'Permeability':[features[7]],
                    'total_Clay':[features[8]],
                    'active_clay':[features[9]],
                    'Dead_clay_':[features[10]],
                    'V_M_':[features[11]],
                    'LOI':[features[12]]
                }
    # print(features)

    condition = analomalies(features)

    if condition != 1:
        final_features = pd.DataFrame(data=dictionary)
        
        # final_features = xgb.DMatrix(final_features.values)

        prediction = model.predict(final_features)

        output = prediction
    # output = len(features)

    else:
        output = 1


    if output == 0:
        output = 'Non-defective mould'
    else:
        output = 'Defective mould'


    return render_template('index.html', prediction_text='Prediction:  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)