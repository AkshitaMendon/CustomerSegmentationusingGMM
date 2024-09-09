from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pickle
import numpy as np
import os

app=Flask(__name__)

#Load Data and Preprocess
data=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%201/customers.csv')
scaler=StandardScaler()
X=scaler.fit(data).transform(data)
PCA3=PCA(n_components=3)
reduced_3_PCA=PCA3.fit(X).transform(X)

#Fit GMM
model=GaussianMixture(n_components=4, random_state=0)
PCA3_pred=model.fit(reduced_3_PCA).predict(reduced_3_PCA)

@app.route('/')
def home():
    return send_from_directory(os.path.join(app.root_path, 'template'), 'predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    json_data=request.get_json()
    new_data=np.array(json_data['data']).reshape(1,-1)
    new_data=scaler.transform(new_data)
    reduced_data=PCA3.transform(new_data)
    prediction=model.predict(reduced_data)
    cluster_data = {
        'x': reduced_3_PCA[:, 0].tolist(),
        'y': reduced_3_PCA[:, 1].tolist(),
        'z': reduced_3_PCA[:, 2].tolist(),
        'cluster': PCA3_pred.tolist()
    }
    return jsonify({'cluster': int(prediction[0]), 'clusterData':cluster_data})

if __name__=='__main__':
    app.run(debug=True)

#Save the scaler and model
with open('SS.pkl','wb') as f:
    pickle.dump(scaler,f)
with open('PCA3.pkl', 'wb') as f:
    pickle.dump(PCA3, f)
with open('gmm_model.pkl', 'wb') as f:
    pickle.dump(model, f)