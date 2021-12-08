
from flask import Flask, render_template, jsonify, request
from joblib import load
import numpy as np
app = Flask(__name__)


@app.route('/', methods=["POST", "GET"])
def getClass():

    # Importe notre algo
    algo = load("algo.joblib")

    # Recuperation du texte de l'utilisateur
    if request.method == "POST":
        # Recuperation du texte de l'utilisateur
        tweet = request.form["tweet"]
        print(type(tweet))
        X = np.array([tweet])
        # algo traite le tweet
        getclass = algo.predict(X)
        return render_template('classRes.html', data=str(getclass[0]))
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
