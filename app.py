import flask
import numpy as np
import sklearn
import joblib
from flask import request, render_template

app = flask.Flask(__name__, template_folder='templates' ,static_folder='static' )

from flask_cors import CORS  #for hosting
CORS(app)


@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/predict' , methods=['POST'])
def predict():
    
    model = joblib.load('iris_flower_prediction_model.ml')
    a = [float(x) for x in request.form.values()]
    arr = [np.array(a)]
    flower_predict = model.predict(arr)
    return render_template('index.html',prediction_text='Predicted Class of Flower: {}'.format(flower_predict))

if __name__ == "__main__" :
    app.run(debug=True)


#Run this file by command py app.py after going into the project directory
# http://127.0.0.1:5000/ this port will get activated showing our REST-API has been created using Flask
