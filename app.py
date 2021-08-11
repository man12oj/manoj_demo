from flask import Flask, request, jsonify, render_template, session

import numpy as np  
from tensorflow.keras.models import load_model
import pickle



def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    
    flower = [[s_len,s_wid,p_len,p_wid]]
    
    flower = scaler.transform(flower)
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    class_ind = model.predict_classes(flower)
    
    return classes[class_ind][0]



app = Flask(__name__)

flower_model = load_model("final_iris_model.h5")
flower_scaler = pickle.load(open("iris_scaler.pkl", "rb))


@app.route('/prediction', methods = ['POST'])
def prediction():

    content = {}

    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['pet_wid'])

    results = return_prediction(model=flower_model,scaler=flower_scaler,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)