from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

model = load_model('../Models/mon_modele.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Image à partir de 'POST'
        image_file = request.files['image']
        # Lecture image 
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        # Redimensionnement image 
        image = cv2.resize(image, (150, 150))
        # Normalisation pixels
        image = image / 255.0
        # Aplatissement image
        image = np.reshape(image, (1, 150, 150, 1))
        
        prediction = model.predict(image)
        if prediction > 0.5:
            result = 'PNEUMONIA'
        else:
            result = 'NORMAL'
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)