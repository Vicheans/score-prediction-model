from flask import Flask, render_template, request
# from flask_assets import Environment, Bundle
import marks as m
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

app = Flask(__name__)


@app.route("/", methods=['GET','POST'])
def marks():
    marks_pred = 0
    hours = 0
    if request.method == "POST":
        hours = request.form["hours"]
        print(hours)
        marks_pred = m.marks_prediction(hours)
        print(marks_pred)

    return render_template("index.html", hrs = hours, mark = marks_pred)

# @app.route


@app.route("/submit", methods = ['POST'])
def submit():
    if request.method == "POST":
        name = request.form["username"]

    return render_template("result.html", name = name)

# def marks_prediction(hours):
#     # marks = marks
#       X = pd.read_csv("https://drive.google.com/file/d/1-3nGY3hyIG4Ki9GO0WEjdd6mUgrEazI1")
#       Y = pd.read_csv("https://drive.google.com/file/d/1-2KBFGkKnSL0mcV4AQ3FhlDgUxwD7gHE")

#       X = X.values
#       Y = Y.values

#       model = LinearRegression()
#       model.fit(X, Y)

    
#       X_test = np.array(hours)
#       X_test = X_test.reshape((1, -1))

#       return model.predict(X_test)[0]


if __name__ == "__main__":
    app.run(debug=True)
