from flask import Flask, request, render_template, jsonify
import pickle
import joblib

# Load the model and vectorizer
model = pickle.load(open('models/my_model.pkl', 'rb'))
vectorizer = joblib.load('models/vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the JSON data sent from the frontend
        data = request.get_json()
        message = data['text']
        
        # Vectorize the input text
        vect = vectorizer.transform([message]).toarray()
        
        # Predict sentiment
        prediction = model.predict(vect)
        
        # Determine sentiment result
        if prediction[0] == 1:
            result = 'Positive'
        else:
            result = 'Negative'
        
        # Return JSON response
        return jsonify({'sentiment': result})

if __name__ == '__main__':
    app.run(debug=True)
