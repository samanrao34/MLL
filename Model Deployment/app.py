#import libraries 
import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle

app= Flask (__name__)
model = pickle.load(open('svc_trained_model.pkl','rb'))

@app.route('/')
def home():
   return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
   inputs= [] 

   inputs.append(request.form['pclass'])
   inputs.append(request.form['gender'])
   inputs.append(request.form['siblings'])
   inputs.append(request.form['embarked'])

   final_inputs=[np.array(inputs)]
   prediction=model.predict(final_inputs)

   if (prediction[0]==1):
    return render_template('index.html', predicted_result= 'Survived')
   if (prediction[0]==0):
    return render_template('index.html', predicted_result= 'Not Survived')

   if __name__ == "__main__":
        app.run(debug = True)