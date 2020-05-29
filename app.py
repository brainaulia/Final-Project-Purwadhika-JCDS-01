from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/item_segmentation', methods=["POST", "GET"])
def cancer_predict():
    if request.method == "POST":
        input = request.form
        feature = [
            float(input['Rating']),
            float(input['Review']),
            float(input['Size']),
            float(input['Installs']),
            float(input['Price']),
            float(input['Last Updated(days ago)'])
        ]

        predcluster = kmean.predict([feature])
        
        endresult = f"Predicted Cluster: {predcluster}"


        return render_template('result.html',
        data=input, prediction=endresult, Rating=input['Rating'],
        Review=input['Review'], Size=input['Size'],
        Installs=input['Installs'], Price=input['Price'],
        Last_Updated=input['Last Updated(days ago)'])

if __name__ == '__main__':
    kmean = joblib.load('cluster_app')
    

    app.run(debug=True, port=4000)