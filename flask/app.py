from flask import Flask, render_template, url_for, json
from flask import jsonify
from flask import request
import pandas as pd
#import tweak
#import auto_arima
#from auto_arima import *
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("main.html")

#Warning: clear the cache of your browser please
@app.route("/graphs")
def graphs():
    return render_template("dash.html")


@app.route("/GetData")
def GetData():
    df = pd.read_csv("result.csv")
    temp = df.to_dict('records')
    columnNames = df.columns.values
    return render_template('table.html', records=temp, colnames=columnNames)


if __name__ == '__main__':
    app.run(debug=True)


