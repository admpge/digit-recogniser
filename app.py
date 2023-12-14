from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import base64
import re

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('mnist_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    data = request.get_json(force=True)
    encoded_image = data['image']
    decoded_image = base64.b64decode(re.sub('^data:image/.+;base64,', '', encoded_image))

    # Convert the decoded image to a PIL image
    image = Image.open(io.BytesIO(decoded_image)).convert('L')
    
    # Resize the image to 28x28 (size expected by the MNIST model)
    image = image.resize((28, 28))

    # Convert the image to a numpy array and normalize it
    image_array = np.array(image) / 255.0

    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the model

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Return the prediction
    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)

