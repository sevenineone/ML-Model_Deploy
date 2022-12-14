from flask import Flask, jsonify, request
import pandas as pd
import joblib
import torch
import numpy as np
from torch.autograd import Variable

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def do_prediction():
    json = request.get_json()
    model = joblib.load('model/ff_model.pkl')
    df = pd.DataFrame(json, index=[0])
    X_test = Variable(torch.from_numpy(df.to_numpy())).float()
    y_predict = model(X_test)
    res = f"{torch.argmax(y_predict, dim=1).numpy()[0]}"
    print(res)
    result = {"Predict" : res}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0')