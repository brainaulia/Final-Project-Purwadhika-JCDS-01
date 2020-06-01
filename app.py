from flask import Flask, render_template, url_for, request
from sklearn.preprocessing import MinMaxScaler
import search as sr
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/RecommendedSystem')
def rec_sys():
    return render_template('index2.html')

@app.route('/item_segmentation', methods=["POST", "GET"])
def segment_predict():
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

        scaler = MinMaxScaler()
        scaler.fit([feature])
        feature=scaler.transform([feature])
        predcluster = kmean.predict(feature)
        
        endresult = f"Predicted Cluster: {predcluster}"


        return render_template('result.html',
        data=input, prediction=endresult, Rating=input['Rating'],
        Review=input['Review'], Size=input['Size'],
        Installs=input['Installs'], Price=input['Price'],
        Last_Updated=input['Last Updated(days ago)'])

@app.route('/item_recommendation', methods=["POST", "GET"])
def item_recommendation():
    if request.method == "POST":
        my_favorite=request.form['search']
        itemrc=sr.make_recommendation(model_knn=nn, data=app_mtx_pca_sparse, fav_app=my_favorite,
                   mapper=app_to_idx, n_recommendation=10)
        
        return render_template('result2.html', name=my_favorite ,recitem=itemrc)
        
        

if __name__ == '__main__':
    kmean = joblib.load('cluster_app')
    app_mtx_pca_sparse = joblib.load('items_embed')
    nn=joblib.load('NN')
    app_to_idx=joblib.load('app_to_idx')
    

    app.run(debug=True, port=4000)