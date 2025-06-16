import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template


# load the trained pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# creating the flask app
app = Flask("__name__")

# setting up the homepage route
@app.route("/")
def home():
    return render_template("index.html")

# creating the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "country" : request.form['country'],
            "year" : int(request.form['year']),
            "bank_account" : request.form['bank_account'],
            "location_type" : request.form['location_type'],
            "household_size" :int(request.form['household_size']),
            "age_of_respondent" :int(request.form['age_of_respondent']),
            "gender_of_respondent" : request.form['gender_of_respondent'],
            "relationship_with_head" : request.form['relationship_with_head'],
            "marital_status" : request.form['marital_status'],
            "education_level" : request.form['education_level'],
            "job_type" : request.form['job_type']
        }
        # convert it to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # make prediction
        prediction = model.predict(input_df)[0]
        output = "Yes, access to cellphone is available." if prediction == "Yes" else "No, no access to cellphone."
        return render_template("index.html", prediction=output)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")
    
if __name__ == "__main__":
    app.run(debug = True)
        