from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np

# Create a Flask app
app = Flask(__name__)

# Load the saved model
model = load('big_mart_sale.joblib')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.form
    
    # Extract the input values
    item_identifier = int(data['item_identifier'])
    item_weight = float(data['item_weight'])
    item_fat_content = int(data['item_fat_content'])
    item_visibility = float(data['item_visibility'])
    item_type = int(data['item_type'])
    item_mrp = float(data['item_mrp'])
    outlet_identifier = int(data['outlet_identifier'])
    outlet_establishment_year = int(data['outlet_establishment_year'])
    outlet_size = int(data['outlet_size'])
    outlet_location_type = int(data['outlet_location_type'])
    outlet_type = int(data['outlet_type'])
    
    # Create a numpy array of the input values
    features = np.array([[item_identifier, item_weight, item_fat_content, item_visibility, item_type, 
                          item_mrp, outlet_identifier, outlet_establishment_year, outlet_size, 
                          outlet_location_type, outlet_type]])
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Return the result as a JSON response
    return render_template('index.html', prediction_text='Estimated Sales: ${:.2f}'.format(prediction[0]))

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
