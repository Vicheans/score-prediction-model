import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def marks_prediction(hours):
    # marks = marks
      df = pd.read_csv("student_info.csv")

      df2 = df[df.study_hours.notna()]
      print (df2)
      # df2
      # X = pd.read_csv("study_hours.csv")
      # Y = pd.read_csv("student_marks.csv")
      X = df2.drop('student_marks', axis=1)
      Y = df2.student_marks
      X = X.values
      Y= Y.values

      plt.scatter(X,Y)
      plt.xlabel("study_hours")
      plt.ylabel("student_marks")
      plt.show()

      print(X)
      print("=============================")
      print(Y)

      model = LinearRegression()
      model.fit(X, Y)

      print("=============================")
      print("Hours ", hours)
      X_test = np.array(hours)
      X_test = X_test.reshape((1, -1))

      return model.predict(X_test)[0]
