from flask import Flask, render_template, request
import numpy as np
import keras
import scipy
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('Admission.h5')

# Define a function to preprocess the user input
def preprocess_input(user_input):
    user_input = np.array(user_input).reshape(1, -1)
    return user_input

# Define a function to make a prediction
def make_prediction(user_input):
    user_input = preprocess_input(user_input)
    prediction = model.predict(user_input)
    if prediction >= 0.5:
        return "Congratulations! You have a high chance of admission."
    else:
        return "Sorry, your chance of admission is low."

# Define the home page route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['cgpa']
        prediction = int(user_input)
        return render_template('result.html', prediction=prediction)
    else:
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
