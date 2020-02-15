from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd


def transform_predict(array):
    v = pickle.load(open('vectorizer.pickle', 'rb'))
    filename = 'model.sav'
    enc = v.transform(array).toarray()
    mod = pickle.load(open(filename, 'rb'))
    pd = mod.predict(enc)
    return pd


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    
    if request.method == 'POST':
    	result = request.form
    	v = []
    	k = []

    	for key,val in result.items():
    		v.append(val)
    		k.append(key)
    df = pd.DataFrame(k,v).reset_index()
    df['loan_id'] = range(1, len(df) + 1)
    df.to_csv('outfile.csv')
    l = v[1:-2]
    
    
    lab = [float(i) if i == l[5] else i for i in l]
    jab = [float(i) if i == lab[6] else i for i in lab]
    
    
    x = transform_predict([jab])
    y =['yes' if x[0] == 1 else 'no' for i in x]
    
    return {'predtiction : ': str(y)}#{'predtiction : ': x}

if __name__ == '__main__':
   app.run(debug = True)