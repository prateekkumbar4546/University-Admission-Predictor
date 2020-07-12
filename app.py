# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Linear regression model
filename = 'GRE_model2.pkl'
model1 = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        gre = int(request.form['GRE'])
        toefl = int(request.form['TOEFL'])
        Univ = int(request.form['University'])
        sop = int(request.form['SOP'])
        lor = int(request.form['LOR'])
        cgpa = float(request.form['CGPA'])
        research = float(request.form['Research'])
        #age = int(request.form['age'])
        
        data = np.array([[gre, toefl, Univ, sop, lor, cgpa, research]])
        my_prediction = model1.predict(data)
        my_prediction = my_prediction*100
        my_prediction = my_prediction[0][0]
        if my_prediction<=0:
            my_prediction = 0
            print("negative")
	elif my_prediction>100:
             my_prediction=100
        else: 
            my_prediction = "{:.2f}".format(my_prediction)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
