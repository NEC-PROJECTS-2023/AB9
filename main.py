import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

model1 = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    state_name =str(request.form['state_name'])
    crop_year = int(request.form['crop_year'])
    season = str(request.form['season'])
    crop = str(request.form['crop'])
    area = int(request.form['area'])

    X = [[state_name, crop_year, season, crop, area]]
    my_pre=model1.predict(X)
    r=my_pre[0]

    return render_template('index.html', r=r)


if __name__ == '__main__':
    app.run(debug=True)
